import tensorflow as tf
from random import randint
import numpy as np

class DQN:
    def __init__(self,sess,gamma):
        self.sess=sess
        self.gamma=gamma

    def compute_loss(self,game_batch_data):
        game_data = game_batch_data.get_data()
        for game in game_data:
            steps = len(game)
            for step, step_data in enumerate(game[3:-1]):
                step = step + 3
                game[step]['loss'] = max(game[step + 1]['action_value'])*self.gamma + game[step]['reward']

    def _get_random_batch(self,game_batch_data,percentage):
        game_data = game_batch_data.get_data()
        no_of_games = len(game_data)
        random_batch_data = []
        for game in game_data:
            no_steps = len(game)
            no_random_data = int(no_steps*percentage/100)
            for x in no_random_data:
                random_batch_data.append(game[randint(3,no_steps-1)])
        return random_batch_data

    def train(self,game_batch_data,epoch_factor):
        total_no_step = game_batch_data.get_total_steps()
        no_of_epochs = int(total_no_step*epoch_factor)
        for x in no_of_epochs:
            random_data = self._get_random_batch(game_batch_data,20)
            new_action = []
            input_observation = []
            for data in random_data:
                input_observation.append(data['concatenated_observation'])

                action_value = data['action_value'][:]
                action_value[data['action']] = data['loss']-action_value[data['action']]
                new_action.append(action_value)

            sess.run(train_op, feed_dict = {input:np.array(train_data),updated_action:np.array(new_action)}
