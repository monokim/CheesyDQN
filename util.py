import cv2
import pyautogui
import numpy as np
import time
import torch

def get_screen(screen, size=(96, 96), device = None):
    image = pyautogui.screenshot(region=screen)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, size)
    image = np.ascontiguousarray(image, dtype=np.float32) / 255
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    return image

def get_screen_rect(caption='Minecraft 1.11.2'):
    hwnd = win32gui.FindWindow(None, caption)
    rect = win32gui.GetWindowRect(hwnd)
    return rect

def click_point(pos=None):
    pyautogui.click(pos)
    #pyautogui.mouseDown(x, y)
    #pyautogui.mouseUp()

def press_key(key):
    pyautogui.keyDown(key)
    time.sleep(0.1)
    pyautogui.keyUp(key)
