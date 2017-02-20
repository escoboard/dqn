import tensorflow as tf
from random import randint
import numpy as np


class DQN:
    def __init__(self, sess, gamma):
        self.sess = sess
        self.gamma = gamma

    def compute_loss(self, game_batch_data):
        game_data = game_batch_data.get_data()

        for game in game_data:
            steps = len(game)

            for step, step_data in enumerate(game[3:-1]):
                #print(step_data['action_value'], step_data['action'], step_data['loss'], step_data['timestamp'],step_data['reward'])
                step += 3
                game[step]['loss'] = max(game[step + 1]['action_value']) * self.gamma + game[step]['reward']

    def _get_random_batch(self, game_batch_data, percentage):
        game_data = game_batch_data.get_data()
        no_of_games = len(game_data)
        random_batch_data = []
        for game in game_data:
            no_steps = len(game)
            no_random_data = int(no_steps * percentage / 100)
            for x in range(no_random_data):
                random_batch_data.append(game[randint(4, no_steps - 2)])
        return random_batch_data

    def train(self, game_batch_data, epochs, graph, graph_input, graph_action_value, graph_updated_value,summary_op,tf_writer):
        total_no_step = game_batch_data.get_total_steps()
        no_of_epochs = int(epochs)
        for x in range(no_of_epochs):
            random_data = self._get_random_batch(game_batch_data, 50)
            update_action = []
            input_observation = []
            actions_value = []

            for data in random_data:
                input_observation.append(np.reshape(data['concatenated_observation'],(-1)))
                step_action_value = data['action_value']
                actions_value.append(step_action_value)
                new_value = step_action_value[:]
                #print(step_action_value)
                #print('Action',data['action'])
                new_value[data['action']] += data['loss']
                update_action.append(new_value)
            input_xy = np.array(input_observation)
            l1= np.array(update_action)
            l2=np.array(actions_value)
            print(input_xy.shape)
            print(l1.shape)

            print(l2.shape)

            self.sess.run(graph, feed_dict={graph_input:input_xy , graph_action_value: l1, graph_updated_value: l2})
            summ = self.sess.run(summary_op, feed_dict={graph_input: input_xy, graph_action_value: l1, graph_updated_value: l2})
            tf_writer.add_summary(summ)
