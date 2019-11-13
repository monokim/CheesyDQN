import cv2
import pyautogui
import numpy as np
import time
import random

def is_done():
    image = pyautogui.screenshot(region=(225, 455, 1, 1))
    image = np.array(image)
    pixel = image[0, 0]
    if pixel[0] == 255 and pixel[1] == 255:
        return True
    return False


class PyQwop:
    bef = 0
    def action(self, action):
        action = action + 1
        if action == self.bef:
            action = random.randint(1, 16)
        if action >> 3 >= 1:
            pyautogui.keyDown('q')

        if (action % 8) >> 2 >= 1:
            pyautogui.keyDown('w')

        if (action % 12) >> 1 >= 1:
            pyautogui.mouseDown(805, 335)
            #pyautogui.keyDown('o')

        if (action % 14) >= 1:
            pyautogui.mouseDown(885, 335)
            #pyautogui.keyDown('p')

        pause = 0.05
        time.sleep(pause)
        pyautogui.keyUp('q')
        pyautogui.keyUp('w')
        pyautogui.mouseUp()
        #pyautogui.keyUp('o')
        #pyautogui.keyUp('p')
        self.bef = action
        print(action)

    def evaluate(self):
        if is_done():
            return -1000
        return 0.1

    def is_done(self):
        return is_done()

    def observe(self):
        return self.bef, 0
        #return random.randint(0, 1), random.randint(0, 1)
