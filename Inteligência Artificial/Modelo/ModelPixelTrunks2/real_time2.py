import os
import cv2
import asyncio
import numpy as np
import time
from ultralytics import YOLO
import pyautogui
import pygetwindow as gw
import ctypes
import math

# Configurações gerais
MODEL_PATH = "runs/detect/train4/weights/best.pt"  # Caminho para o modelo treinado
ROBOFLOW_SIZE = 960  # Tamanho das imagens
FRAMERATE = 15  # Taxa de quadros por segundo
SCREEN_CENTER_OFFSET = 100  # Pixels para considerar o centro da tela

# Configurações de automação
TREE_BREAK_TIME = 1.5  # Tempo para quebrar uma árvore
MOVE_DURATION = 0.6  # Duração do movimento em segundos
DETECTION_COOLDOWN = 1  # Tempo de espera após interagir com uma árvore

# Distância da árvore (em % da área da tela)
TREE_MIN_AREA_RATIO = 0.045  # A árvore deve ocupar pelo menos 2% da área da tela
TREE_MAX_AREA_RATIO = 0.07  # Mas não mais que 6% (para não encostar)

CLASS_NAMES = [
    "Acacia_log",
    "Birch_log",
    "Cherry_log",
    "Dark_oak_log",
    "Jungle_log",
    "Oak_log",
    "Spruce_log"
]

# Inicializar o modelo YOLOv8
model = YOLO(MODEL_PATH)

def is_minecraft_open():
    windows = gw.getAllTitles()
    for window in windows:
        if "Minecraft" in window:
            return True
    return False

def capture_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot

# --- Mouse e teclado nativos Windows ---
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_MOVE = 0x0001

def press_key(key_code):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(key_code, 0, 0, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(key_code):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(key_code, 0, 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def move_forward(duration=0.51):
    print("Movendo para frente...")
    press_key(VK_W)
    time.sleep(duration)
    release_key(VK_W)

def move_forward_small(duration=0.18):
    print("Movendo para frente (pequeno)...")
    press_key(VK_W)
    time.sleep(duration)
    release_key(VK_W)

def move_backward(duration=0.2):
    print("Movendo para trás...")
    press_key(VK_S)
    time.sleep(duration)
    release_key(VK_S)

def move_mouse_relative(dx, dy):
    print(f"Movendo mouse: dx={dx}, dy={dy}")
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
    command = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

def click_mouse_down():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def click_mouse_up():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def break_tree():
    print("Quebrando árvore...")
    click_mouse_down()
    time.sleep(TREE_BREAK_TIME)
    click_mouse_up()

def find_nearest_tree(trees, screen_width, screen_height):
    if not trees:
        return None
    center_x, center_y = screen_width // 2, screen_height // 2
    nearest_tree = None
    min_distance = float('inf')
    for tree in trees:
        x1, y1, x2, y2 = tree["box"]
        tree_center_x = (x1 + x2) / 2
        tree_center_y = (y1 + y2) / 2
        distance = math.sqrt((tree_center_x - center_x)**2 + (tree_center_y - center_y)**2)
        if distance < min_distance:
            min_distance = distance
            nearest_tree = tree
    return nearest_tree

def is_tree_centered(tree_box, screen_width, screen_height):
    x1, y1, x2, y2 = tree_box
    tree_center_x = (x1 + x2) / 2
    tree_center_y = (y1 + y2) / 2
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    distance_x = abs(tree_center_x - screen_center_x)
    distance_y = abs(tree_center_y - screen_center_y)
    is_centered = distance_x < SCREEN_CENTER_OFFSET and distance_y < SCREEN_CENTER_OFFSET
    print(f"Centralização: dx={distance_x}, dy={distance_y}, centralizado={is_centered}")
    return is_centered

def is_at_good_distance(tree_box, screen_width, screen_height):
    x1, y1, x2, y2 = tree_box
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    screen_area = screen_width * screen_height
    area_ratio = box_area / screen_area
    is_good = TREE_MIN_AREA_RATIO <= area_ratio <= TREE_MAX_AREA_RATIO
    print(f"Área da árvore: {area_ratio:.2%} da tela (ideal: {TREE_MIN_AREA_RATIO:.2%}-{TREE_MAX_AREA_RATIO:.2%})")
    print(f"Distância boa (por área): {is_good}")
    return is_good

def aim_at_tree(tree_box, screen_width, screen_height):
    x1, y1, x2, y2 = tree_box
    tree_center_x = (x1 + x2) / 2
    tree_center_y = (y1 + y2) / 2
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    
    # Calcular diferença entre centro da árvore e centro da tela
    dx = int(tree_center_x - screen_center_x)
    dy = int(tree_center_y - screen_center_y)
    
    print(f"Ajustando mira: dx={dx}, dy={dy}")
    
    # Ajustar sensibilidade para evitar sobrecorreção
    sensitivity = 0.6
    if abs(dx) < 30 and abs(dy) < 30:
        sensitivity = 0.4  # Mais preciso para ajustes finos
    
    dx = int(dx * sensitivity)
    dy = int(dy * sensitivity)
    
    # Evitar movimentos muito pequenos
    if abs(dx) < 3 and abs(dy) < 3:
        print("Movimento muito pequeno, ignorando")
        return True
    
    # Usar o movimento de mouse do código original
    move_mouse_relative(dx, dy)
    
    # Retornar True se o ajuste foi pequeno
    return abs(dx) < 10 and abs(dy) < 10

def adjust_distance_to_tree(tree_box, screen_width, screen_height):
    x1, y1, x2, y2 = tree_box
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    screen_area = screen_width * screen_height
    area_ratio = box_area / screen_area

    # Faixa para movimento pequeno (exemplo: 3.5% a 4.5%)
    SMALL_STEP_MIN = 0.035
    SMALL_STEP_MAX = 0.045

    if area_ratio > TREE_MAX_AREA_RATIO:
        print(f"Muito perto da árvore ({area_ratio:.2%}), recuando...")
        move_backward()
        return True
    elif SMALL_STEP_MIN <= area_ratio < SMALL_STEP_MAX:
        print(f"Entre 4 e 4,5 blocos ({area_ratio:.2%}), avançando pequeno...")
        move_forward_small()
        return True
    elif area_ratio < SMALL_STEP_MIN:
        print(f"Longe da árvore ({area_ratio:.2%}), avançando normal...")
        move_forward()
        return True
    else:
        print(f"Distância ideal para a árvore ({area_ratio:.2%})")
        return False

async def automate_minecraft():
    def detect_trees(image):
        results = model.predict(source=image, save=False, conf=0.5)
        trees = []
        
        # Better debug info
        print("--- Detection Results ---")
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Accept all valid tree types (not just class_id 0)
                if 0 <= class_id < len(CLASS_NAMES):
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    print(f"Árvore detectada: {CLASS_NAMES[class_id]} (class_id={class_id}) - "
                          f"Coords: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) - Confiança: {confidence:.2f}")
                    trees.append({
                        "box": (x1, y1, x2, y2),
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id]
                    })
                else:
                    print(f"Classe desconhecida: id={class_id} - Confiança: {confidence:.2f}")
        else:
            print("Nenhum objeto detectado no frame")
                
        print(f"Total de árvores encontradas: {len(trees)}")
        return trees

    last_interaction_time = 0
    last_tree_area_ratio = 0

    print("Automação do Minecraft iniciada. Pressione Ctrl+C para sair.")
    print("Aguardando detecção de árvores...")

    while True:
        if not is_minecraft_open():
            print("Minecraft não está aberto. Aguardando...")
            await asyncio.sleep(5)
            continue

        screenshot = capture_screenshot()
        img_height, img_width, _ = screenshot.shape
        print(f"Tamanho da tela: {img_width}x{img_height}")

        trees = detect_trees(screenshot)
        print(f"Árvores detectadas: {len(trees)}")

        current_time = time.time()
        if current_time - last_interaction_time < DETECTION_COOLDOWN:
            await asyncio.sleep(0.1)
            continue

        if trees:
            nearest_tree = find_nearest_tree(trees, img_width, img_height)
            if nearest_tree:
                tree_box = nearest_tree["box"]
                confidence = nearest_tree.get("confidence", 0.0)
                class_name = nearest_tree.get("class_name", "Desconhecido")
                print(f"Árvore mais próxima: {class_name} - {tree_box} - Confiança: {confidence:.2f}")

                x1, y1, x2, y2 = tree_box
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                screen_area = img_width * img_height
                area_ratio = box_area / screen_area
                print(f"Tamanho da bounding box: {box_width}x{box_height} pixels ({area_ratio:.2%} da tela)")

                last_tree_area_ratio = area_ratio

                if not is_tree_centered(tree_box, img_width, img_height):
                    print("Árvore não centralizada, ajustando mira...")
                    aim_at_tree(tree_box, img_width, img_height)
                    last_interaction_time = current_time - DETECTION_COOLDOWN + 0.2
                    continue

                if is_at_good_distance(tree_box, img_width, img_height):
                    print("Árvore centralizada e a uma boa distância! Quebrando...")
                    break_tree()
                    move_backward(0.2)
                    last_interaction_time = current_time
                else:
                    print("Ajustando distância para a árvore...")
                    moved = adjust_distance_to_tree(tree_box, img_width, img_height)
                    if moved:
                        last_interaction_time = current_time
        else:
            print("Nenhuma árvore detectada! Procurando...")

            if last_tree_area_ratio > TREE_MIN_AREA_RATIO:
                print("Árvore desapareceu quando estávamos próximos, tentando quebrar...")
                break_tree()
                last_tree_area_ratio = 0
                last_interaction_time = current_time
                continue

            if current_time - last_interaction_time > DETECTION_COOLDOWN * 2:
                print("Girando para procurar árvores...")
                move_mouse_relative(100, 0)
                last_interaction_time = current_time - DETECTION_COOLDOWN + 0.5
                last_tree_area_ratio = 0

        await asyncio.sleep(0.1)

async def main():
    print("Iniciando automação do Minecraft...")
    print("Pressione Ctrl+C para encerrar o programa")
    try:
        await automate_minecraft()
    except KeyboardInterrupt:
        print("Programa encerrado pelo usuário")

if __name__ == "__main__":
    asyncio.run(main())