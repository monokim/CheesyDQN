import gym
from gym import spaces
import numpy as np
from gym_game.envs.pygame import PyGame

class GameEnv(gym.Env):
    metadata = {'render.modes' : ['human']}
    def __init__(self):
        print("init")
        self.action_space = spaces.Discrete(16)
        #self.observation_space = spaces.Box(low=0, high=255, shape=(16, 16), dtype=np.int)
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([15, 1]), dtype=np.int)
        self.pygame = PyGame()
        self.memory = []

    def reset(self):
        del self.pygame
        pyautogui.mouseDown(500, 150)
        pyautogui.mouseUp()
        pyautogui.press('space')
        self.pygame = PyGame()
        obs = self.pygame.observe()
        return obs

    def step(self, action):
        self.pygame.action(action)
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        obs = self.pygame.observe()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        pass

    def save_memory(self, file):
        np.save(file, self.memory)
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
