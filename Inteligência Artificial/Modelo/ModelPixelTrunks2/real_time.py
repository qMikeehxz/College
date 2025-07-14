import os
import cv2
import asyncio
import numpy as np
import time
from ultralytics import YOLO
import pyautogui
import pygetwindow as gw

# Configurações gerais
MODEL_PATH = "runs/detect/train4/weights/best.pt"  # Caminho para o modelo treinado
ROBOFLOW_SIZE = 960  # Tamanho das imagens
FRAMERATE = 15  # Taxa de quadros por segundo

# Inicializar o modelo YOLOv8
model = YOLO(MODEL_PATH)

# Função para verificar se o Minecraft está aberto
def is_minecraft_open():
    windows = gw.getAllTitles()
    for window in windows:
        if "Minecraft" in window:
            return True
    return False

# Função para capturar uma captura de tela
def capture_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot

# Função para inferência em tempo real com a webcam
async def infer_webcam():
    video = cv2.VideoCapture(1)  # Use a webcam
    last_frame = time.time()

    while True:
        if cv2.waitKey(1) == ord('q'):  # Pressione 'q' para sair
            break

        ret, img = video.read()
        if not ret:
            print("Erro ao acessar a webcam.")
            break

        height, width, channels = img.shape
        scale = ROBOFLOW_SIZE / max(height, width)
        img = cv2.resize(img, (round(scale * width), round(scale * height)))

        results = model.predict(source=img, save=False, conf=0.5)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), (230, 143, 69), 2)
            cv2.putText(img, f"{confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 143, 69), 2)

        cv2.imshow("Inferencia em Tempo Real", img)

        elapsed = time.time() - last_frame
        await asyncio.sleep(max(0, 1 / FRAMERATE - elapsed))
        last_frame = time.time()

    video.release()
    cv2.destroyAllWindows()

# Função para automação no Minecraft
async def automate_minecraft():
    def detect_trees(image):
        results = model.predict(source=image, save=False, conf=0.5)
        trees = []
        for box in results[0].boxes:
            if box.cls[0] == 0:  # Classe 0 corresponde a "trunk_tree"
                trees.append({
                    "x": (box.xyxy[0][0] + box.xyxy[0][2]) / 2,
                    "y": (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                })
        return trees

    def move_and_break_tree(tree, screen_width=3072, screen_height=1920):
        x = tree["x"]
        y = tree["y"]
        screen_x = int(x * screen_width)
        screen_y = int(y * screen_height)

        pyautogui.moveTo(screen_x, screen_y)
        pyautogui.mouseDown(button='left')
        time.sleep(5)
        pyautogui.mouseUp(button='left')

        print(f"Quebrando madeira na posição ({screen_x}, {screen_y})!")

    while True:
        if not is_minecraft_open():
            print("Minecraft não está aberto. Aguardando...")
            await asyncio.sleep(5)
            continue

        screenshot = capture_screenshot()
        trees = detect_trees(screenshot)
        if trees:
            move_and_break_tree(trees[0])
        else:
            print("Nenhuma árvore detectada!")
        await asyncio.sleep(1)

# Função principal para escolher a tarefa
async def main():
    print("Escolha a funcionalidade:")
    print("1 - Inferência em tempo real com a webcam")
    print("2 - Automação no Minecraft")
    choice = input("Digite o número da funcionalidade desejada: ")

    if choice == "1":
        await infer_webcam()
    elif choice == "2":
        await automate_minecraft()
    else:
        print("Escolha inválida. Encerrando o programa.")

if __name__ == "__main__":
    asyncio.run(main())