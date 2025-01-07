import os
import csv
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tqdm import tqdm
import mediapipe as mp  # MediaPipe para segmentação

# Configurações
IMAGE_DIR = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegmentTrain/dataset/images"
OUTPUT_CSV = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegmentTrain/dataset/annotations/dados_faces.csv"
MASKS_DIR = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegmentTrain/dataset/newMasks"
MODEL_PATH = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegmentTrain/dataset/outputs/best_model.keras"

def load_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise ValueError("Erro: Não foi possível carregar o Haarcascade!")
    return face_cascade

def exclude_face_regions(hair_mask, faces, margin=100):
    mask_no_faces = hair_mask.copy()
    for (x, y, w, h) in faces:
        mask_no_faces[y + h:, :] = 0  # Zera abaixo do rosto
        x_start = max(x + int(0.0 * w), 10)
        y_start = max(y + int(0.1 * h), 3)
        x_end = min(x + w - int(0.0 * w), mask_no_faces.shape[1])
        y_end = min(y + h + int(0.1 * h), mask_no_faces.shape[0])
        mask_no_faces[y_start:y_end, x_start:x_end] = 0  # Zera ao redor do rosto
    return mask_no_faces

def refine_mask(mask, min_area=40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    refined_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Ignora o fundo
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            refined_mask[labels == i] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    return refined_mask

def save_mask(mask, filename, masks_directory):
    mask_filename = masks_directory / f"{filename.stem}.png"
    cv2.imwrite(str(mask_filename), mask)

import os

def process_image(image_file, face_cascade):
    # Lê a imagem
    image = cv2.imread(str(image_file))
    if image is None:
        print(f"Erro ao carregar {image_file}, ignorando.")
        return None, None

    # Conversão para escala de cinza e detecção de faces
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=2, minSize=(50, 50))

    # Inicializa o MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image_rgb)
        hair_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

    # Define o caminho do diretório para salvar as máscaras
    output_dir = "HairSegmentTrain/dataset/newMasks"
    os.makedirs(output_dir, exist_ok=True)  # Cria o diretório, se não existir

    # Salva as máscaras intermediárias e final no diretório correto
    base_filename = os.path.splitext(os.path.basename(image_file))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_mask_raw.png"), hair_mask)

    # Exclui regiões do rosto e salva a máscara sem rostos
    hair_mask_no_faces = exclude_face_regions(hair_mask, faces, margin=50)  # Reduz a margem para 50
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_mask_no_faces.png"), hair_mask_no_faces)

    # Refina a máscara e salva a máscara final
    refined_hair_mask = refine_mask(hair_mask_no_faces, min_area=500)  # Reduz min_area para 500
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_mask_refined.png"), refined_hair_mask)

    return refined_hair_mask, image.shape[:2]


def detect_faces_and_hair(images_dir, output_csv, masks_dir):
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
