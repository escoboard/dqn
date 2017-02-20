import gym

env = gym.make('Breakout-v0')
env.reset()
for _ in range(10):
    env.render()
    env.step(env.action_space.sample())
    print(env.action_space.sample())