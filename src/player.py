import tensorflow as tf
import src.environment as environment
import src.graph as Graph
import src.dqn as dqn
import src.utils as utils
import src.game_data as game_data
from random import randint
import numpy as np

class Player:
    def __int__(self):
        self.no_of_steps = 1000
        self.game_train_batch_size = 10
        self.env = environment.GymEnvironment('Breakout-v0')
        self.sess = tf.Session()
        self.graph,self.graph_input,self.graph_action_value,self.graph_updated_action,self.graph_train_op = Graph.Graph(self.env.actions(),self.sess)
        self.dqn = dqn.DQN(self.sess,gamma=0.7)



    def main(self):

        print('Exection started at %d'%utils.get_timestamp())


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
                four_obervation_list = []
                while not done:
                    step_timestamp = utils.get_timestamp(True)
                    no_of_steps +=1
                    if no_of_steps <5:
                        action = randint(0,self.env.actions()-1)

                    else:
                        action_value = self.sess.run(
                            self.graph_action_value,
                            feed_dict = {self.graph_action_value:concatenated_input})
                        concatenated_input = np.concatenate((four_obervation_list[3], four_obervation_list[2],
                                                             four_obervation_list[1], four_obervation_list[0]), axis=1)

                    action = np.argmax(action_value)
                    obv,reward,done,info = self.env.step(action)

                    if no_of_steps >4:
                        four_obervation_list.pop(0)
                    #pop a element from the four_obervation_list
                    four_obervation_list.append(obv)

                    # put data to the store
                    batch_data_store.add_step(step_timestamp,obv,concatenated_input,reward,action_value,action,self.graph_train_op)
            #computing loss for all the dataset
            self.dqn.compute_loss(batch_data_store)
            self.dqn.train(batch_data_store,1000)
            batch_data_store.save_progress()





