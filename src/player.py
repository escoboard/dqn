from random import randint, random

import numpy as np
import tensorflow as tf

import dqn as dqn
import environment as environment
import game_data as game_data
import graph as graph
import utils as utils
from PIL import Image


class Player:
    def __init__(self):
        self.no_of_steps = 1000000000000
        self.game_train_batch_size = 3
        self.env = environment.GymEnvironment('Pong-v0')
        self.sess = tf.Session()
        self.G = graph.Graph(self.env.actions(), self.sess)
        self.graph, self.graph_input, self.graph_action_value, self.graph_updated_action ,self.loss= self.G.get_graph()
        self.dqn = dqn.DQN(self.sess, gamma=0.8)
        self.sess.run(tf.global_variables_initializer())
        self.tf_merged_summary_op = tf.summary.merge_all()
        self.tf_writer = tf.summary.FileWriter('output', self.sess.graph)

    def play(self):

        print('Execution started at %d with %d' % (utils.get_timestamp(),self.env.actions()))
        epsilon = 0.1
        for x in range(self.no_of_steps):
            print('epsilon',epsilon)
            batch_timestamp = utils.get_timestamp(True)
            batch_data_store = game_data.GameBatchData(batch_timestamp)
            print('Batch %d started' % x)
            for y in range(self.game_train_batch_size):
                print (x,y)

                game_timestamp = utils.get_timestamp(True)
                print('game %d started at %d'%(y,game_timestamp))
                batch_data_store.new_game(game_timestamp)
                done = False
                action = 0
                action_value = np.zeros((10))
                self.env.reset()
                no_of_steps = 0
                concatenated_input = None
                four_observation_list = []
                stacked_input = None
                while not done:
                    step_timestamp = utils.get_timestamp(True)
                    no_of_steps += 1
                    if no_of_steps < 5:
                        action = randint(0, self.env.actions() - 1)

                    else:
                        stacked_input = np.dstack(four_observation_list)
                        concatenated_input = np.array([stacked_input])
                        action_value = self.sess.run(
                            self.graph_action_value,
                            feed_dict={self.graph_input:concatenated_input})
                        action_value=action_value[0]
                        action = np.argmax(action_value)
                        #print(action_value)

                    epsilon = self.exploration_probability(epsilon, 0.9, 0.000001)
                    if random() > epsilon:
                        # Random action
                        action = randint(0, self.env.actions() - 1)
                    obv, reward, done, info = self.env.step(action)

                    if no_of_steps > 4:
                        four_observation_list.pop(0)
                    # pop a element from the four_observation_list
                    four_observation_list.append(np.array(Image.fromarray(obv).convert('L')))

                    # put data to the store
                    batch_data_store.add_step(step_timestamp, obv, stacked_input, reward, action_value, action)
            # computing loss for all the dataset
            print ("Batch %d done training started"%x)
            self.dqn.compute_loss(batch_data_store)
            self.dqn.train(batch_data_store, 100 , self.graph, self.graph_input, self.graph_action_value,
                           self.graph_updated_action,self.tf_merged_summary_op,self.tf_writer,self.loss)
            batch_data_store.save_progress()
            self.G.save_graph('output/'+str(batch_timestamp)+'.ckpt')
            print("Batch %d done training done" % x)

    def exploration_probability(self, current, maximum, increment):
        current += increment
        return min(maximum,current)


if __name__ == '__main__':
    player = Player()
    player.play()
