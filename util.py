import cv2
import pyautogui
import numpy as np
import time
import torch

def get_screen(x = 0, y = 0, w = 1920, h = 1080, size=(96, 96)):
    image = pyautogui.screenshot(region=(x, y, w, h))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, size)
    image = np.ascontiguousarray(image, dtype=np.float32) / 255
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image

def click_point(x, y):
    pyautogui.mouseDown(x, y)
    pyautogui.mouseUp()

def press_key(key):
    pyautogui.keyDown(key)
    time.sleep(0.1)
    pyautogui.keyUp(key)
