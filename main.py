import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Definição de Constantes e Caminhos
CSV_PATH = '/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegment/dataset/annotations/dados_faces.csv'
IMAGE_DIR = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegment/dataset/images"
MASK_DIR = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegment/dataset/newMasks"
OUTPUT_DIR = "/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegment/dataset/outputs"
IMAGE_SIZE = (256, 256)  # Tamanho aumentado para melhorar a qualidade da saída
EPOCHS = 200
BATCH_SIZE = 16  # Ajuste para otimizar o uso de memória

def validate_coordinates(row: pd.Series) -> bool:
    coords = row[['xmin', 'ymin', 'xmax', 'ymax']]
    return all((coord is not None and coord >= 0 for coord in coords)) and row['xmax'] > row['xmin'] and row['ymax'] > row['ymin']

def preprocess_image_and_mask(image: np.ndarray, mask: np.ndarray) -> tuple:
    mask_resized = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)  # Interpolação bilinear para suavizar
    mask_binary = (mask_resized > 0).astype(np.uint8)
    image_resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return image_resized, mask_binary

def load_annotations() -> pd.DataFrame:
    try:
        annotations = pd.read_csv(CSV_PATH)
        required_columns = {'filename', 'xmin', 'ymin', 'xmax', 'ymax'}
        if not required_columns.issubset(annotations.columns):
            raise ValueError(f"O CSV deve conter as colunas: {', '.join(required_columns)}")
        return annotations
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return pd.DataFrame()

def load_data() -> tuple:
    annotations = load_annotations()
    if annotations.empty:
        print("Nenhuma anotação encontrada.")
        return None, None

    X, y = [], []
    for _, row in annotations.iterrows():
        image_path = os.path.join(IMAGE_DIR, row['filename'])
        mask_filename = row['filename'].replace('.jpg', '.png')
        mask_path = os.path.join(MASK_DIR, mask_filename)

        if not all(os.path.exists(p) for p in [image_path, mask_path]):
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        image_resized, mask_resized = preprocess_image_and_mask(image, mask)
        X.append(image_resized)
        y.append(mask_resized)

    return np.array(X), np.array(y, dtype=np.uint8)

def augment_image_and_mask(image, mask):
    # Expande a dimensão do mask para ter pelo menos três dimensões
    mask = tf.expand_dims(mask, axis=-1)
    
    # Aplica as transformações desejadas
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Remove a dimensão extra depois de flip para manter a compatibilidade com o restante do pipeline
    mask = tf.squeeze(mask, axis=-1)

    return image, mask

def create_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.map(lambda x, y: augment_image_and_mask(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def build_unet(input_shape: tuple) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)(p3)  # Dilatação para maior contexto
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)

    # Upsampling
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return models.Model(inputs=[inputs], outputs=[outputs])

# Fluxo Principal
X, y = load_data()
if X is None or y is None or len(X) == 0:
    print("Erro ao carregar dados: Nenhum dado válido encontrado.")
else:
    # Verifica se há dados suficientes para divisão
    if len(X) < 2:
        print(f"Aviso: Número insuficiente de exemplos ({len(X)}). Usando todos para treinamento.")
        train_dataset = create_tf_dataset(X, y).cache().prefetch(tf.data.AUTOTUNE)
        val_dataset = None
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        train_dataset = create_tf_dataset(X_train, y_train).cache().prefetch(tf.data.AUTOTUNE)
        val_dataset = create_tf_dataset(X_val, y_val, shuffle=False)

    model = build_unet(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=990, restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, 'best_model.keras'), save_best_only=True),
        ReduceLROnPlateau(patience=990)
    ]

    # Treinamento com ou sem validação
    if val_dataset is not None:
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)
    else:
        history = model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)

    # Conversão para TFLite com quantização em Float16
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    tflite_model_path = os.path.join(OUTPUT_DIR, "model.tflite")
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print("Modelo treinado e salvo com sucesso.")