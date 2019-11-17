import cv2
import pyautogui
import numpy as np
import time
import torch
import torchvision.transforms as T

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40),
                    T.ToTensor()])

def get_screen(x = 0, y = 0, w = 1920, h = 1080, size=(640, 360)):
    image = pyautogui.screenshot(region=(x, y, w, h))
    image = np.array(image)
    resized_image = cv2.resize(image, size)
    screen = np.ascontiguousarray(resized_image, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)

def click_point(x, y):
    pyautogui.mouseDown(x, y)
    pyautogui.mouseUp()

def press_key(key):
    pyautogui.keyDown(key)
    time.sleep(0.1)
    pyautogui.keyUp(key)
