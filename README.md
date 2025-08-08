# Classificador de Embalagens de Fio Dental (Cheia ou Vazia)

## 📖 Descrição do Projeto

Este projeto apresenta uma solução de Visão Computacional proposta como trabalho final do **bootcamp de machine learning**.

O modelo implementado realiza a classificação binária de embalagens de fio dental, determinando se elas estão **cheias** ou **vazias**. 

Uma aplicação plausível seria automatizar o processo de verificação de integridade do produto em uma linha de produção ou estoque, utilizando uma Rede Neural Convolucional (CNN) treinada para a tarefa.

O sistema foi desenvolvido em Python, utilizando a biblioteca TensorFlow/Keras para a construção e treinamento do modelo, no ambiente online do Google Colab.

<img width="1133" height="356" alt="image" src="https://github.com/user-attachments/assets/bc6c04d2-bd38-4818-89cd-728a5f648938" />
<img width="1159" height="337" alt="image" src="https://github.com/user-attachments/assets/73dffcd5-451a-4f02-a04b-a10677a0d06f" />


## 🛠️ Funcionalidades

-   **Classificação Binária:** Distingue as embalagens entre cheias e vazias.
-   **Dataset Robusto:** Utiliza um dataset com mais de 4000 imagens capturadas manualmente em condições variadas.
-   **Pré-processamento Automatizado:** Inclui scripts para processar as imagens originais e prepará-las para o treinamento.


## 📁 Estrutura do Projeto

```
.
├── misc/                      # Scripts do pipeline de pré-processamento do dataset.
│   ├── 01-organizar_DS.py
│   ├── 02-selecionar_ROI.py
│   └── 03-ajustar_DS.py
├── card28.ipynb               # Código com o modelo 
├── CNN-Classifier-Model.h5    # Modelo salvo
├── dataset_dental_floss.zip   # Dataset
├── predict_offline.py         # Código para execução do modelo salvo
├── README.md
└── requirements.txt
 
```
 
## 📊 O Dataset

A base de dados do projeto é um dataset criado especificamente para este problema.

-   **Total de Imagens:** 4066
-   **Classes:** `Cheia` e `Vazia`
-   **Origem:** Fotos tiradas manualmente com um smartphone, garantindo uma variedade de ângulos, iluminações e posicionamentos.

<p align="center">
  <img width="264" height="353" alt="image" src="https://github.com/user-attachments/assets/3dfe1d88-744a-44be-9bb7-6c592e92ee1c" />
  &nbsp;&nbsp;&nbsp;&nbsp;    <img width="260" height="320" alt="image" src="https://github.com/user-attachments/assets/5bad0042-9ae1-4f57-af1a-b885aa171146" />
 
</p>


### Etapas de Pré-processamento

Para garantir que o modelo recebesse dados adequados, as imagens originais passaram por um **pipeline de pré-processamento** dividido em três etapas principais, utilizando os scripts localizados na pasta `misc/`.

**1. Organizando as Imagens do Dataset (`misc/01-organizar_DS.py`)**

Este script foi o ponto de partida para a estruturação dos dados. Ele foi responsável por organizar as imagens originais, separando-as em suas respectivas pastas de classe (`cheia`/`vazia`) 

```sh
$ python3.12 01-organizar_DS.py --pasta_entrada /caminho/das/fotos/originais/do/dataset/ --pasta_saida /caminho/das/fotos/renomedas/por/classe --prefixo nome-da-classe
```


**2. Selecionando as Região de Interesse (ROI) (`misc/02-selecionar_ROI.py`)**

As fotos originais continham muito ruído de fundo. Para que o modelo focasse exclusivamente na embalagem, este script foi utilizado para cortar a Região de Interesse (ROI) de cada imagem.

```sh
$ python3.12 02-selecionar_ROI.py --pasta_entrada /caminho/das/fotos/renomeadas --pasta_saida /caminho/das/regiões/de/interesse/recortadas
```

<p align="center">
  <img width="230" height="341" alt="image" src="https://github.com/user-attachments/assets/947456f8-fb04-4357-b0d5-8d5c56917833" />
  &nbsp;&nbsp;&nbsp;&nbsp;    
  <img width="260" height="341" alt="image" src="https://github.com/user-attachments/assets/45c84c62-e48b-4dcd-84a4-3839c05ce139" />
  &nbsp;&nbsp;&nbsp;&nbsp;    
  <img width="193" height="193" alt="image" src="https://github.com/user-attachments/assets/7c24756d-f644-400c-80a7-bfd16c96afa5" />

</p>

**3. Redimensionamento e Convertendo para Cinza  (`misc/03-ajustar_DS.py`)**

A etapa final de preparação. Este script processa as imagens cortadas para:
-   **Redimensionar:** Todas as imagens foram padronizadas para as dimensões exigidas pela entrada do modelo (128x128 pixels).
-   **Conversão da Escala de Cores:** As imagens foram todas convertidas para escala de cinza com o objetivo de otimizar e acelerar o processo de treinamento da rede neural.

```sh
python3.12 03-ajustar_DS.py --pasta_entrada /caminho/das/regiões/de/interesse/recortadas --pasta_saida /caminho/do/dataset/final
```

<p align="center">
  <img width="293" height="293" alt="image" src="https://github.com/user-attachments/assets/7c24756d-f644-400c-80a7-bfd16c96afa5" />
  &nbsp;&nbsp;&nbsp;&nbsp;    
  <img width="142" height="146" alt="image" src="https://github.com/user-attachments/assets/e215ec80-3892-4b3e-b3dd-c2e06eee326f" />

</p>

## 🧠 O Modelo de IA

O cerébro deste projeto é uma **Rede Neural Convolucional (CNN)**, com 3 camadas convolucionais.

### Arquitetura

O modelo foi construído com a biblioteca TensorFlow/Keras com a seguinte estrutura:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 1)),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Camada de Dropout para regularização
    tf.keras.layers.Dropout(dropout_rate),
    
    # Camada de saída com 2 neurônios (um para cada classe)
    tf.keras.layers.Dense(num_classes) 
])
```

### Performance

Após o treinamento com os hiperparâmetros inicialmente propostos foram feitos testes utilizando a técnica de **Grid Search** para otimizá-los.
Por fim, os valores utilizados no modelo final foram:

- **Taxa de Aprendizagem (Learning Rate):** 1e-3
- **Taxa de Dropout:** 0.4
- **Épocas de Treinamento:** 13
- **Tamanho do Lote (Batch Size):** 32

A matriz de confusão abaixo ilustra o desempenho do modelo:

<p align="center">
  <img width="519" height="417" alt="image" src="https://github.com/user-attachments/assets/302a5f4c-95a2-41df-97c7-476d03d77e4d" />
</p>

## 🚀 Como Usar

### Pré-requisitos para execução offline

-   Python 3.8+
-   Tensorflow
-   Numpy
-   OpenCV
    
### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/augustolnb/Dental-Floss-Empty-Full-Classifier-CNN.git
    cd /caminho/do/repositorio/Dental-Floss-Empty-Full-Classifier-CNN
    ```

2.  **Crie e ative um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

### Executando uma Previsão

Para realizar previsões, é possível usar o modelo treinado **CNN-Classifier-Model.h5** com o script **predict-offline.py**.
Basta modificar no código o caminho das imagens que gostaria de classificar.

```sh
$ python3.12 predict_offline.py

```
Exemplo de saída:

<p align='center'>
  <img width="402" height="162" alt="image" src="https://github.com/user-attachments/assets/d3e48eaa-5a1f-4e19-88bd-a977fbd88083" />
</p>


## ✒️ Autor

**Lucas Augusto Nunes de Barros**

-   **LinkedIn:** `[https://www.linkedin.com/in/augusto-lnb/]`
-   **GitHub:** `[https://github.com/augustolnb]`

