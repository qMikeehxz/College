from IPython import display
display.clear_output()

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import roboflow

import ultralytics
from ultralytics import YOLO
ultralytics.checks()

import torch

# Verificar se o backend MPS está disponível (para Apple Silicon)
print("MPS disponível:", torch.backends.mps.is_available())

# Fazer login no Roboflow
# Certifique-se de que a pasta de configuração do Roboflow tem permissões adequadas (MacOS):
# - sudo mkdir ~/.config/roboflow (criar pasta onde vai guardar a conf.)
# - sudo chown -R $USER:staff ~/.config/roboflow (dar permissões ao seu usuário)
roboflow.login()

# Importar a chave da API do arquivo api_key.py
from api_key import api_key

# Inicializar o Roboflow com a chave da API
rf = roboflow.Roboflow(api_key)

# Listar os projetos disponíveis no workspace
print("Listando projetos disponíveis no workspace...")
workspace = rf.workspace("artificial-intelligence-l0rp4")  # Nome do workspace
print("Projetos disponíveis no workspace:")
for project in workspace.projects():
    print(f"- {project}")  # Apenas imprima o nome do projeto diretamente

# Substituir pelo nome do workspace e do projeto no Roboflow
workspace_name = "artificial-intelligence-l0rp4"  # Nome do workspace
project_name = "pixelvision2"  # Nome do projeto
project = rf.workspace(workspace_name).project(project_name)

# Listar as versões disponíveis do dataset
print("Listando versões disponíveis do dataset...")
for version in project.versions():
    print(f"- Versão: {version.id}, Nome: {version.name}")

# Substituir pela versão correspondente do dataset
dataset_version = 2 # Versão do dataset (ajuste se necessário)
dataset = project.version(dataset_version).download("yolov8")

# Verificar os paths no arquivo data.yaml após o download
print("Dataset baixado. Verifique os paths no arquivo 'data.yaml'.")

# Carregar o modelo pré-treinado YOLOv8
# Lista de modelos disponíveis: https://docs.ultralytics.com/models/yolov8/#performance-metrics
model = YOLO("yolov8m.pt")  # Modelo médio (YOLOv8m)

# Treinar o modelo
# Substituir 'FoE-bot-2/data.yaml' pelo caminho correto do arquivo data.yaml
# Ajustar o número de épocas (epochs) e o tamanho das imagens (imgsz) conforme necessário
results = model.train(
    data='PixelVision2-2/data.yaml',  # Caminho correto para o arquivo data.yaml
    epochs=100,
    imgsz=640,
    device='cpu'
)

# Após o treinamento, os resultados estarão disponíveis na pasta 'runs/train'
print("Treinamento concluído. Verifique os resultados na pasta 'runs/train'.")