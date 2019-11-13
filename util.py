import cv2
import pyautogui
import numpy as np
import time

def get_screen(x = 0, y = 0, w = 1920, h = 1080, size=(640, 360)):
    image = pyautogui.screenshot(region=(x, y, w, h))
    image = np.array(image)
    resized_image = cv2.resize(image, size)
    return resized_image

def click_point(x, y):
    pyautogui.mouseDown(x, y)
    pyautogui.mouseUp()

def press_key(key):
    pyautogui.keyDown(key)
    time.sleep(0.1)
    pyautogui.keyUp(key)
