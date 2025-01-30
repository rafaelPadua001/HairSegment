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
MARGIN = 1

mp_face_detection = mp.solutions.face_detection

def load_face_detector():
    """Carrega o modelo Haarcascade para detecção de faces."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise ValueError("Erro: Não foi possível carregar o Haarcascade!")
    return face_cascade


def exclude_face_regions(
        hair_mask, 
        faces,
        margin=0,
        width_reduction=0.15,
        forehead_ratio=0.01,
        neck_expansion=10,
        preserve_hair=True,
        face_height_reduction=0.87):
    """
    Exclui regiões do rosto da máscara de cabelo usando uma elipse para melhor ajuste.
    """
    mask_no_faces = hair_mask.copy()

    for (x, y, w, h) in faces:
        # Calcula a margem lateral para excluir as orelhas
        left_margin = int(w * width_reduction)
        x_start_face = max(0, x + left_margin)
        x_end_face = min(mask_no_faces.shape[1], x + w - left_margin)

        # Define a área da testa (franja)
        forehead_height = int(h * forehead_ratio)
        y_start_forehead = max(0, y - forehead_height)
        y_end_forehead = y  

        # Suaviza a transição na região da testa
        forehead_region = mask_no_faces[y_start_forehead:y_end_forehead, x_start_face:x_end_face]
        for col in range(forehead_region.shape[1]):
            column_data = forehead_region[:, col]
            hair_top_idx = np.argmax(column_data > 0)  # Encontra o topo do cabelo

            if hair_top_idx > 0:  # Se o cabelo estiver presente na coluna
                gradient = np.linspace(1, 0, len(column_data[hair_top_idx:]))
                forehead_region[hair_top_idx:, col] *= gradient[:len(forehead_region[hair_top_idx:])]

        # Atualiza a máscara com a testa suavizada
        mask_no_faces[y_start_forehead:y_end_forehead, x_start_face:x_end_face] = forehead_region

        # Cria uma máscara em branco para desenhar a elipse
        ellipse_mask = np.zeros_like(mask_no_faces)
        
        # Define os parâmetros da elipse
        center = (x + w // 2, y + h // 2)  # Centro do rosto
        axes = (int(w * 0.5), int(h * 0.7))  # Semi-eixos da elipse (ajuste conforme necessário)
        angle = 0  # Ângulo de rotação da elipse

        # Desenha a elipse na máscara
        cv2.ellipse(ellipse_mask, center, axes, angle, 0, 360, 255, -1)

        # Exclui a região da elipse da máscara de cabelo
        mask_no_faces[ellipse_mask > 0] = 0

        # Expansão do pescoço (opcional)
        neck_with_expansion = int(w * neck_expansion)
        x_start_neck = max(0, x - neck_with_expansion)
        x_end_neck = min(mask_no_faces.shape[1], x + w + neck_with_expansion)
        y_end_body = mask_no_faces.shape[0]  # Até o final da imagem
        mask_no_faces[y_end_forehead:y_end_body, x_start_neck:x_end_neck] = 0

    return mask_no_faces

def exclude_shadow_regions(image, hair_mask, faces, shadow_threshold=0.9):
    """
    Exclui regiões escuras (sombras) dentro das áreas do rosto detectadas.
    
    Args:
        image (numpy.ndarray): Imagem original em escala de cinza ou BGR.
        hair_mask (numpy.ndarray): Máscara binária do cabelo.
        faces (list): Coordenadas (x, y, w, h) dos rostos detectados.
        shadow_threshold (int): Limite de intensidade para considerar como sombra (0 a 255).
    
    Returns:
        numpy.ndarray: Máscara sem as regiões de sombra.
    """
    shadow_free_mask = hair_mask.copy()
    
    # Certifique-se de que a imagem está em escala de cinza
    if len(image.shape) == 3:  # Imagem colorida
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    for (x, y, w, h) in faces:
        # Define a região do rosto
        face_region = gray_image[y:y+h, x:x+w]
        hair_region = hair_mask[y:y+h, x:x+w]

        # Cria uma máscara para as sombras na região do rosto
        shadow_mask = cv2.inRange(face_region, 0, shadow_threshold)

        # Remove sombras da máscara do cabelo
        hair_region[shadow_mask > 0] = 0

        # Atualiza a máscara geral
        shadow_free_mask[y:y+h, x:x+w] = hair_region

    return shadow_free_mask

def refine_mask(mask, min_area=2):
    """
        Refina a máscara removendo ruídos e pequenas áreas.
        
        Args:
            mask (numpy.ndarray): Máscara binária a ser refinada.
            min_area (int): Tamanho mínimo de área para manter na máscara.
        
        Returns:
        numpy.ndarray: Máscara refinada.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    refined_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):  # Ignora o fundo
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            centroid_y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] // 2
            if centroid_y < mask.shape[0] * 0.5:
                refined_mask[labels == i] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel maior para suavização
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return refined_mask

def save_mask(mask, filename, masks_directory):
    """Salva a máscara de cabelo no diretório especificado."""
    mask_filename = masks_directory / f"{filename.stem}.png"
    cv2.imwrite(str(mask_filename), mask)

def detect_faces_mediapipe(image):
    """
    Detecta rostos em uma imagem usando MediaPipe Face Detection.

    Args:
        image (numpy.ndarray): Imagem em formato BGR.

    Returns:
        list: Lista de coordenadas (x, y, w, h) dos rostos detectados.
    """
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                faces.append((x, y, w, h))
        return faces
    
def process_image(image_file):
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

    # Inicializa o MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image_rgb)
        hair_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

    # Detecta rostos usando o MediaPipe Face Detection
    faces = detect_faces_mediapipe(image)

    # Exclui regiões do rosto e partes inferiores
    hair_mask_no_faces = exclude_face_regions(hair_mask, faces, margin=0, width_reduction=0.15, forehead_ratio=0.01)

    # Refina a máscara
    refined_hair_mask = refine_mask(hair_mask_no_faces, min_area=2)

    return refined_hair_mask, (image.shape[1], image.shape[0])

def detect_faces_and_hair(images_dir, output_csv, masks_dir):
    """Processa as imagens para detectar cabelo e rostos, salvando os resultados."""
    if not os.path.isdir(images_dir):
        raise ValueError(f"Diretório de imagens '{images_dir}' não encontrado!")

    # face_cascade = load_face_detector()
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

        image_files = list(Path(images_dir).rglob('*.[jp][pn]*'))
        for image_file in tqdm(image_files, desc="Processando imagens"):
            refined_mask, dimensions = process_image(image_file)
            if refined_mask is None:
                continue

            save_mask(refined_mask, image_file, masks_dir)
            logging.info(f"Máscara salva para {image_file}")

            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                writer.writerow([image_file.name, dimensions[1], dimensions[0], 'hair', x, y, x + w, y + h])

if __name__ == "__main__":
    detect_faces_and_hair(IMAGE_DIR, OUTPUT_CSV, MASKS_DIR)