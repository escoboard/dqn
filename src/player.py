import tensorflow as tf
import src.environment as environment
import src.graph as Graph
import src.dqn as dqn
import src.utils as utils
import src.game_data as game_data
from random import randint, random
import numpy as np


class Player:
    def __int__(self):
        self.no_of_steps = 1000
        self.game_train_batch_size = 10
        self.env = environment.GymEnvironment('Breakout-v0')
        self.sess = tf.Session()
        self.graph, self.graph_input, self.graph_action_value, self.graph_updated_action = Graph.Graph(
            self.env.actions(), self.sess)
        self.dqn = dqn.DQN(self.sess, gamma=0.7)
        self.sess.run(tf.global_variables_initializer())

    def play(self):

        print('Execution started at %d' % utils.get_timestamp())
        epsilon = 0.1
        for x in range(self.no_of_steps):
            batch_timestamp = utils.get_timestamp(True)
            batch_data_store = game_data.GameBatchData(batch_timestamp)

            for y in range(self.game_train_batch_size):
                game_timestamp = utils.get_timestamp(True)
                batch_data_store.new_game(game_timestamp)
                done = False
                action = 0
                action_value = np.zeros((10))
                self.env.reset()
                no_of_steps = 0
                concatenated_input = None
                four_observation_list = []
                while not done:
                    step_timestamp = utils.get_timestamp(True)
                    no_of_steps += 1
                    if no_of_steps < 5:
                        action = randint(0, self.env.actions() - 1)

                    else:
                        concatenated_input = np.concatenate((four_observation_list[3], four_observation_list[2],
                                                             four_observation_list[1], four_observation_list[0]),
                                                            axis=1)
                        action_value = self.sess.run(
                            self.graph_action_value,
                            feed_dict={self.graph_input: concatenated_input})

                        action = np.argmax(action_value)
                    epsilon = self.exploration_probability(epsilon, 0.9, 0.000001)
                    if epsilon > random():
                        # Random action
                        action = randint(0, self.env.actions() - 1)
                    obv, reward, done, info = self.env.step(action)

                    if no_of_steps > 4:
                        four_observation_list.pop(0)
                    # pop a element from the four_observation_list
                    four_observation_list.append(obv)

                    # put data to the store
                    batch_data_store.add_step(step_timestamp, obv, concatenated_input, reward, action_value, action)
            # computing loss for all the dataset
            self.dqn.compute_loss(batch_data_store)
            self.dqn.train(batch_data_store, 30, self.graph, self.graph_input, self.graph_action_value, self.graph_updated_action)
            batch_data_store.save_progress()

    def exploration_probability(self, current, maximum, increment):
        current += increment
        if current > maximum:
            return maximum

if __name__ == '__main__':
    player = Player()
    player.play()