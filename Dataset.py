import os
import csv
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tqdm import tqdm
import mediapipe as mp
import logging

# Configurações globais
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
IMAGE_DIR = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegment/dataset/images"
OUTPUT_CSV = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegment/dataset/annotations/dados_faces.csv"
MASKS_DIR = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegment/dataset/newMasks"
MIN_AREA = 5
MARGIN = 1

def load_face_detector():
    """Carrega o modelo Haarcascade para detecção de faces."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise ValueError("Erro: Não foi possível carregar o Haarcascade!")
    return face_cascade

def exclude_face_regions(hair_mask, faces, margin=1, width_reduction=0.95, preserve_hair=True):
    """
    Exclui regiões do rosto e partes inferiores da máscara de cabelo, ajustando ao formato do rosto.
    - hair_mask: máscara de cabelo original.
    - faces: lista de rostos detectados.
    - margin: margem ao redor do rosto (vertical).
    - width_reduction: proporção para reduzir a largura da região excluída.
    - preserve_hair: preserva a parte superior do cabelo.
    """
    mask_no_faces = hair_mask.copy()
    
    for (x, y, w, h) in faces:
        # Ajuste horizontal: reduz a largura da região excluída
        left_margin = int(w * width_reduction / 1)
        print('Left Margin', left_margin)
        x_start_face = max(10, x + left_margin)
        x_end_face = min(mask_no_faces.shape[1], x + w - left_margin)

        # Ajuste vertical: preserva a parte superior do cabelo, se necessário
        if preserve_hair:
            y_start_face = max(0, y + int(h * 0))  # Exclui apenas abaixo de 30% da altura do rosto
        else:
            y_start_face = max(0, y)

        # Parte inferior do rosto com margem
        y_end_face = min(y + h + margin, mask_no_faces.shape[0])

        # Exclui a região ajustada do rosto
        mask_no_faces[y_start_face:y_end_face, x_start_face:x_end_face] = 0
        
        # Exclui tudo abaixo da parte inferior do rosto
        mask_no_faces[y_end_face:, :] = 0
    
    return mask_no_faces


def refine_mask(mask, min_area=MIN_AREA):
    """
    Refina a máscara de cabelo removendo ruídos e pequenas áreas, com controle melhorado.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    refined_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Ignora o fundo
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            refined_mask[labels == i] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Kernel maior para suavização
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return refined_mask

def save_mask(mask, filename, masks_directory):
    """Salva a máscara de cabelo no diretório especificado."""
    mask_filename = masks_directory / f"{filename.stem}.png"
    cv2.imwrite(str(mask_filename), mask)

def process_image(image_file, face_cascade):
    """
    Processa a imagem, detecta rosto e cabelo, e refina as máscaras.
    """
    # Lê a imagem e redimensiona para um tamanho padrão para consistência
    image = cv2.imread(str(image_file))
    if image is None:
        logging.warning(f"Erro ao carregar {image_file}, ignorando.")
        return None, None

    # Redimensiona a imagem para uma altura padrão (exemplo: 512px)
    height, width = image.shape[:2]
    scale_factor = 512 / height
    image = cv2.resize(image, (int(width * scale_factor), 512), interpolation=cv2.INTER_AREA)

    # Conversão para escala de cinza e detecção de faces
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Inicializa o MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image_rgb)
        hair_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

    # Exclui regiões do rosto e partes inferiores
    hair_mask_no_faces =  exclude_face_regions(hair_mask, faces, margin=2, width_reduction=0.1)

    # Refina a máscara
    refined_hair_mask = refine_mask(hair_mask_no_faces)

    return refined_hair_mask, (image.shape[1], image.shape[0])

def detect_faces_and_hair(images_dir, output_csv, masks_dir):
    """Processa as imagens para detectar cabelo e rostos, salvando os resultados."""
    if not os.path.isdir(images_dir):
        raise ValueError(f"Diretório de imagens '{images_dir}' não encontrado!")

    face_cascade = load_face_detector()
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

        image_files = list(Path(images_dir).rglob('*.[jp][pn]*'))
        for image_file in tqdm(image_files, desc="Processando imagens"):
            refined_mask, dimensions = process_image(image_file, face_cascade)
            if refined_mask is None:
                continue

            save_mask(refined_mask, image_file, masks_dir)

            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                writer.writerow([image_file.name, dimensions[1], dimensions[0], 'hair', x, y, x + w, y + h])

if __name__ == "__main__":
    detect_faces_and_hair(IMAGE_DIR, OUTPUT_CSV, MASKS_DIR)
