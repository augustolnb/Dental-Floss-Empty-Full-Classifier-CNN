import tensorflow as tf
import numpy as np
import os
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--caminho_modelo', 
    type=str, 
    required=True, 
    help='Caminho para o arquivo do modelo (.h5).'
)

parser.add_argument(
    '--pasta_imagens', 
    type=str, 
    required=True, 
    help='Caminho para a pasta contendo as imagens para inferência.'
)

args = parser.parse_args()

CLASS_NAMES = ['empty', 'full']

try:
    model = tf.keras.models.load_model(args.caminho_modelo)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

caminho_abs_pasta = os.path.expanduser(args.pasta_imagens)
if not os.path.isdir(caminho_abs_pasta):
    print(f"A pasta de imagens '{caminho_abs_pasta}' não foi encontrada.")
    exit()

for nome_arquivo in os.listdir(caminho_abs_pasta):
    caminho_imagem = os.path.join(caminho_abs_pasta, nome_arquivo)
    
    if not os.path.isfile(caminho_imagem):
        continue

    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Não foi possível carregar a imagem: {nome_arquivo}")
        continue
    
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype("float32") / 255.0
    
    img_tensor = np.expand_dims(img_normalized, axis=-1)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    predictions = model.predict(img_tensor, verbose=0)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_class_index]
    
    probabilities = tf.nn.softmax(predictions).numpy()[0]
    
    print(f"\n-----------------------------")
    print(f"Imagem: {nome_arquivo} -> Classe Predita: {predicted_class}")
    print(f"  Probabilidades: {CLASS_NAMES[0]}: {probabilities[0]:.2f}, {CLASS_NAMES[1]}: {probabilities[1]:.2f}")
    
print("\nInferência concluída.")
