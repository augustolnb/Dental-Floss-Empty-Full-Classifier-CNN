import tensorflow as tf
import numpy as np
import os
import cv2

# Defina o caminho para o seu modelo e para a pasta de imagens
MODELO_CAMINHO = 'CNN-Classifier-Model.h5'
PASTA_IMAGENS = '~/Público/dataset-final/prediction/'

# Defina a ordem das classes, a mesma que foi usada no treinamento
CLASS_NAMES = ['empty', 'full']

# 1. Carregar o modelo
try:
    model = tf.keras.models.load_model(MODELO_CAMINHO)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()


# 3. Pré-processar e prever cada imagem na pasta
caminho_abs_pasta = os.path.expanduser(PASTA_IMAGENS)
if not os.path.isdir(caminho_abs_pasta):
    print(f"A pasta de imagens '{caminho_abs_pasta}' não foi encontrada.")
    exit()

for nome_arquivo in os.listdir(caminho_abs_pasta):
    caminho_imagem = os.path.join(caminho_abs_pasta, nome_arquivo)
    
    # Ignorar arquivos que não sejam imagens (ex: .DS_Store no macOS)
    if not os.path.isfile(caminho_imagem):
        continue

    # 4. Pré-processamento da imagem
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Não foi possível carregar a imagem: {nome_arquivo}")
        continue
    
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype("float32") / 255.0
    
    # Ajustar as dimensões para o formato (1, 128, 128, 1)
    img_tensor = np.expand_dims(img_normalized, axis=-1)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # 5. Fazer a predição
    predictions = model.predict(img_tensor, verbose=0)
    
    # Obter a classe com maior probabilidade
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_class_index]
    
    # Obter as probabilidades (logits convertidos)
    probabilities = tf.nn.softmax(predictions).numpy()[0]
    
    # 6. Imprimir o resultado
    print(f"\n-----------------------------")
    print(f"Imagem: {nome_arquivo} -> Classe Predita: {predicted_class}")
    print(f"  Probabilidades: {CLASS_NAMES[0]}: {probabilities[0]:.2f}, {CLASS_NAMES[1]}: {probabilities[1]:.2f}")
    
print("\nInferência concluída.")
