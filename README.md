# Classificador de Embalagens de Fio Dental

## Observação Importante!!
Devido ao tamanho dos arquivos, tanto o dataset final - **dataset-dental-floss.zip (512x512-RGB)** -  como o **notebook python (.ipynb)** estão hospedados no Google Drive e podem ser acessados através do link: https://drive.google.com/drive/folders/1IrVB9KGnZcH3Ao1z7g0gTmWubeltKvBD?usp=sharing

## Descrição do Projeto

Este projeto apresenta uma solução de Visão Computacional proposta como trabalho final do **bootcamp de machine learning** do grupo de pesquisa LAMIA.

O modelo implementado realiza a classificação binária de embalagens de fio dental, determinando se elas estão **cheias** ou **vazias**. 

Uma aplicação plausível seria automatizar o processo de verificação do produto em uma linha de produção ou estoque, utilizando uma Rede Neural Convolucional (CNN) treinada para a tarefa.

O sistema foi desenvolvido em Python, utilizando a biblioteca TensorFlow/Keras para a construção e treinamento do modelo, no ambiente online do Google Colab.

<img width="1133" height="356" alt="image" src="https://github.com/user-attachments/assets/bc6c04d2-bd38-4818-89cd-728a5f648938" />


## Organização do Projeto

```
.
├── data-pipeline/                 # Scripts do pipeline de pré-processamento do dataset.
│   ├── 01-organizar_DS.py
│   ├── 02-selecionar_ROI.py
│   └── 03-ajustar_DS.py
├── Relatorio28-LucasAugusto-V4.pdf       
├── card28_V4.xlsx                           # Arquivo com dados de testes
├── modelo_cinza-final.h5                    # Modelo cinza salvo
├── modelo_rgb-final.h5                      # Modelo rgb salvo  
├── dataset_dental_floss(128x128-GRAY).zip   # Dataset testado inicialmente
├── predict_offline.py                       # Código para execução do modelo salvo
├── README.md
└── requirements.txt
 
```
 
## Base de dados

A base de dados do projeto é um dataset criado especificamente para este problema.

-   **Total de Imagens:** 4066
-   **Classes:** `Cheia` e `Vazia`
-   **Origem:** Fotos tiradas manualmente com um smartphone, garantindo uma variedade de ângulos, iluminações e posicionamentos.
-   **Divisão:** O dataset foi divido em 75% das amostras para treinamento, 15% para validação e 10% para teste.
-   **Técnicas:** O conjunto de treinamento foi aumentando artificialmente através da técnica de Data Augmentation.
<br>
<p align="center">
  <img width="264" height="353" alt="image" src="https://github.com/user-attachments/assets/3dfe1d88-744a-44be-9bb7-6c592e92ee1c" />
  &nbsp;&nbsp;&nbsp;&nbsp;    <img width="260" height="320" alt="image" src="https://github.com/user-attachments/assets/5bad0042-9ae1-4f57-af1a-b885aa171146" />
 
</p>


### Etapas de Pré-processamento Inicial (128x128-GRAY)

Para garantir que o modelo recebesse dados adequados, as imagens originais passaram por um **pipeline de pré-processamento** dividido em três etapas principais, utilizando os scripts localizados na pasta `data_pipeline/`.

**1. Organizando as Imagens do Dataset (`data_pipeline/01-organizar_DS.py`)**

Este script foi o ponto de partida para a estruturação dos dados. Ele foi responsável por organizar as imagens originais, separando-as em suas respectivas pastas de classe (`cheia`/`vazia`) 

```sh
$ python3.12 01-organizar_DS.py --pasta_entrada /caminho/das/fotos/originais/do/dataset/ --pasta_saida /caminho/das/fotos/renomedas/por/classe --prefixo nome-da-classe
```


**2. Selecionando as Região de Interesse (ROI) (`data_pipeline/02-selecionar_ROI.py`)**

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

**3. Redimensionamento e Convertendo para Cinza  (`data_pipeline/03-ajustar_DS.py`)**

A etapa final de preparação. Este script processa as imagens cortadas para:
-   **Redimensionar:** Todas as imagens foram padronizadas para as dimensões exigidas pela entrada do modelo (128x128 pixels).
-   **Converter a Escala de Cores:** As imagens foram todas convertidas para escala de cinza com o objetivo de otimizar e acelerar o processo de treinamento da rede neural.

```sh
python3.12 03-ajustar_DS.py --pasta_entrada /caminho/das/regiões/de/interesse/recortadas --pasta_saida /caminho/do/dataset/final
```

<p align="center">
  <img width="293" height="293" alt="image" src="https://github.com/user-attachments/assets/7c24756d-f644-400c-80a7-bfd16c96afa5" />
  &nbsp;&nbsp;&nbsp;&nbsp;    
  <img width="142" height="146" alt="image" src="https://github.com/user-attachments/assets/e215ec80-3892-4b3e-b3dd-c2e06eee326f" />

</p>

## Pré-processamento Final

Ao fim dos primeiros testes, o dataset com as images originais foi processados novamente, com o mesmo pipeline de dados, porém com novas dimensões e uma nova escala de cores, afim de permitir uma variabilidade maior nos testes, com o objetivo de melhorar as métricas do modelo.

- O dataset final esta armazenado no Google Drive devido ao seu tamanho.
- Esse dataset possui imagens RGB com dimensões 512x512

## Os Modelos de Rede Neural

- Para o projeto foram desenvolvidas duas **Redes Neurais Convolucionais (CNN)**, uma contendo 4 e outra contendo 5 camadas convolucionais. 
- Para a definição da estrutura e dos hiperparâmetros foram aplicadas as técnicas de **Otimização Bayesiana** e **Grid Search**.

### Arquiteturas

Os modelos foram construídos com a biblioteca TensorFlow/Keras, as estruturas são apresentadas abaixo.

#### Modelo Alimentado por Imagens RGB

Hiperparâmetros:

- **Taxa de Aprendizagem (Learning Rate):** 4.66428e-4
- **Taxa de Dropout:** 0.25
- **Neurônios na Camada Densa:** 192
- **Tamanho do Lote (Batch Size):** 32

```python
final_model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(224, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(112, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(192, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(num_classes)
])
```

#### Modelo Alimentado por Imagens em Escala de Cinza

Hiperparâmetros:

- **Taxa de Aprendizagem (Learning Rate):** 7e-05
- **Taxa de Dropout:** 0.3
- **Neurônios na Camada Densa:** 112
- **Tamanho do Lote (Batch Size):** 32

```python
final_model = keras.Sequential([
    layers.Input(shape=(128, 128, 1)),

    layers.Conv2D(160, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(144, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(240, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(112, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(112, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes)
])
```

### Resultados

O dataset final foi utilizado para refinar o modelo, aumentando a possibilidade de diferentes combinações de parâmetros no processamento dos dados de entrada, juntamente com diferentes configurações de **data augmentation**. Um dos modelos é alimentado com imagens de dimensões 128x128 em RGB, enquanto o outro utiliza imagens de mesmas dimensões mas em escala de cinza.

Para ambos os modelos, apesar das variações testadas, foram utilizadas as seguintes configurações de **data augmentation**:

- Fator de Contraste: 20%
- Espelhamento horizontal 

### Modelo RGB

#### Gráficos de acurácia e perda durante o treino e a validação:

<p align="center">
<img width="755" height="373" alt="image" src="https://github.com/user-attachments/assets/937d1dc7-7467-4645-beed-35d4b2afb8e7" />
</p>

#### Matriz de confusão.

O modelo obteve taxa de acerto de 90,23% quando analisado com o conjunto de teste.

<p align="center">
 <img width="308" height="316" alt="image" src="https://github.com/user-attachments/assets/d9a2a46b-999f-4708-a7c8-8fd91bee7e09" />

</p>

#### Curva ROC

O modelo obteve valor de AUC = 0.9782 e a respectiva curva ROC é apresentada a seguir:

<p align="center">
<img width="314" height="316" alt="image" src="https://github.com/user-attachments/assets/69f2f408-ef96-4df0-9e3d-cda3bf042562" />
</p>

### Modelo Cinza

#### Gráficos de acurácia e perda durante o treino e a validação:

<p align="center">
  <img width="784" height="365" alt="image" src="https://github.com/user-attachments/assets/ee564ca9-51dd-47e4-849b-d2bb721a9705" />
</p>

#### Matriz de confusão.

O modelo obteve taxa de acerto de 87,5% quando analisado com o conjunto de teste.

<p align="center">
  <img width="308" height="316" alt="image" src="https://github.com/user-attachments/assets/3ba4b3de-8673-41e8-aa19-4fee184108c6" />
</p>

#### Curva ROC

O modelo obteve valor de AUC = 0.9372 e a respectiva curva ROC é apresentada a seguir:

<p align="center">
<img width="314" height="316" alt="image" src="https://github.com/user-attachments/assets/a2fc0dd8-85a3-42d2-94da-02c80b8c282a" />
</p>


## Fazendo Previsões

- Além da possibilidade de utilizar o modelo diretamente a partir do Google Colab, também é possível baixar o arquivo do modelo para fazer classificações localmente.

### Pré-requisitos para execução local

-   Python 3.8+
-   Tensorflow
-   Numpy
-   OpenCV
    
### Instalação

1.  **Clonando o repositório:**
    ```bash
    git clone https://github.com/augustolnb/Dental-Floss-Empty-Full-Classifier-CNN.git
    cd /caminho/do/repositorio/Dental-Floss-Empty-Full-Classifier-CNN
    ```

2.  **Criando e ativando um ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Instalando as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

### Executando a Rede

Para realizar previsões, é possível usar o modelo treinado **model14_final.h5** com o script **predict-offline.py**.
Basta passar como argumento do código o caminho para o modelo salvo e para a pasta das imagens que gostaria de classificar.

```sh
$ python3.12 predict-offline.py --caminho_modelo /caminho/do/modelo/salvo/model14_final.h5 --pasta_imagens /caminho/das/imagens

```
<br>

## Predições com imagens inéditas:

### Modelo RGB
<p align='center'>
  <img width="828" height="817" alt="image" src="https://github.com/user-attachments/assets/522a0ead-9271-4f7c-8033-600ac69a94c7" />
</p>

### Modelo Cinza
<p align='center'>
  <img width="829" height="822" alt="image" src="https://github.com/user-attachments/assets/f395c3c6-e116-43e0-9d5c-5c412f1ae181" />
</p>


## ✒️ Autor

**Lucas Augusto Nunes de Barros**

-   **LinkedIn:** `[https://www.linkedin.com/in/augusto-lnb/]`
-   **GitHub:** `[https://github.com/augustolnb]`

