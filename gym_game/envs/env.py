import gym
from gym import spaces
import numpy as np
from gym_game.envs.game import Game

class Env(gym.Env):
    metadata = {'render.modes' : ['human']}
    def __init__(self):
        ###############################################
        # modify here!!!
        self.action_space = spaces.Discrete(16)
        ###############################################
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([15, 1]), dtype=np.int)
        self.game = Game()
        self.memory = []

    def reset(self):
        del self.game
        self.game = Game()
        obs = self.game.observe()
        return obs

    def step(self, action):
        self.game.action(action)
        reward = self.game.evaluate()
        done = self.game.is_done()
        obs = self.game.observe()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.game.view()
        pass

    def save_memory(self, file):
        np.save(file, self.memory)
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
