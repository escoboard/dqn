import gym


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

    def make(self):
        self.env = gym.make(self.game_name)

    def reset(self):
        self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        return self.env.step(action)

    def action_space(self):
        return self.env.action.space
