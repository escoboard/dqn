import random

from utils import get_timestamp
from game_data import GameBatchData
from environment import GymEnvironment


def test_game_data():
    g = GameBatchData(get_timestamp(True))
    env = GymEnvironment('Atlantis-v0')
    env.reset()
    for _ in range(5):
        i = g.new_game(get_timestamp(True))
        print("Game: %s" % i)
        for _ in range(5):
            observation, reward, done, info = env.step(0)
            d = g.add_step(
                timestamp=get_timestamp(True),
                observation=observation,
                concatenated_observation=observation,
                reward="rew%s" % (random.randint(1, 10)),
                action_value=["av%s" % (random.randint(1, 10)), "av%s" % (random.randint(1, 10))],
                action="a%s" % (random.randint(1, 10)),
            )
            print("Step %s" % d)
    g.save_progress()

if __name__ == '__main__':
    test_game_data()
