import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = load_model('/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegmentTrain/dataset/outputs/best_model.keras')

# Caminho para a pasta de imagens de teste
test_images_path = '/home/rafael/Área de Trabalho/HairSegmentationTrain/HairSegmentTrain/dataset/testeImage'

# Função para pré-processar as imagens
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Ler a imagem
    img = cv2.resize(img, (256, 256))  # Redimensionar para o tamanho que o modelo espera (758x758)
    img = img / 255.0  # Normalizar os pixels para a faixa [0, 1]
    return img

# Listar todas as imagens no diretório de teste
test_images = [os.path.join(test_images_path, img) for img in os.listdir(test_images_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Listas para armazenar resultados
predictions = []
file_names = []

# Loop pelas imagens de teste
for image_path in test_images:
    img = preprocess_image(image_path)  # Pré-processar a imagem
    img_expanded = np.expand_dims(img, axis=0)  # Adicionar a dimensão do lote
    prediction = model.predict(img_expanded)  # Fazer a previsão
    predictions.append(prediction)  # Armazenar a previsão
    file_names.append(os.path.basename(image_path))  # Armazenar o nome do arquivo

# Exibir resultados
for file_name, prediction in zip(file_names, predictions):
    # Verificar a forma e os valores da previsão
    print(f"Previsão (forma): {prediction.shape}, valores: {prediction.flatten()[:10]}")  # Mostra os primeiros 10 valores
    print(f"Média da previsão: {np.mean(prediction)}, Máximo da previsão: {np.max(prediction)}")

    # Aplicar threshold para obter a máscara binária
    binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)  # Ajustar threshold se necessário

    # Ler a imagem original para visualização
    original_img = cv2.imread(os.path.join(test_images_path, file_name))
    original_img_resized = cv2.resize(original_img, (256, 256))  # Redimensionar a imagem original

    # Mostrar a imagem original e a máscara prevista lado a lado
    plt.figure(figsize=(15, 5))

    # Exibir imagem original
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')

    # Exibir saída do modelo sem threshold
    plt.subplot(1, 3, 2)
    plt.imshow(prediction[0, :, :, 0], cmap='gray')  # Mostrar a saída do modelo
    plt.title('Saída do Modelo (sem threshold)')
    plt.axis('off')

    # Exibir máscara prevista
    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask, cmap='gray')  # Mostrar a máscara em escala de cinza
    plt.title('Máscara Prevista')
    plt.axis('off')

    plt.show()
