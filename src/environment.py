import gym
import numpy as np
from PIL import Image


class Environment():
    def __init__(self):
        self.game_name = None
        self.env = None

    def make(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def action_space(self):
        raise NotImplementedError


class GymEnvironment(Environment):
    def __init__(self, game_name):
        super().__init__()
        self.game_name = game_name
        self.make()

    def make(self):
        self.env = gym.make(self.game_name)

    def reset(self):
        self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.array(Image.fromarray(observation).resize((84, 84)))
        return observation, reward, done, info

    def actions(self):
        return self.env.action_space.n


def main():
    env = GymEnvironment('MsPacman-v0')
    env.reset()
    no_of_action = env.actions()
    for _ in range(1000):
        env.render()
        observation, reward, done, info = env.step(0)
        print(observation)


if __name__ == '__main__':
    main()
