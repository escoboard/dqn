import src.dqn_new as dqn
import unittest
import numpy as np
class TestDQN(unittest.TestCase):
    def __init__(self):
        self.agent= dqn.Agent()


    def test_build_network(self):
        print("Testing network building")
        s, q_values, model = self.agent.build_network()
        self.trainable_weights = model.trainable_weights
        print(s)
        print(q_values)
        print(model)


    def test_build_training_op(self):
        print("Testing network building training operation")
        a, y, loss, grads_update = self.agent.build_training_op(self.trainable_weights)
        print(a)
        print(y)
        print(loss)
        print(grads_update)


    def test_get_initial_state(self):
        print("Testing fetching initial state")
        stack = self.agent.get_initial_state(np.array([100,100]))
        print(stack)


    def test_get_action(self):
        print("Testing getting action")
        action, action_value = self.agent.get_action(np.array([100,100]))
        print(action)
        print(action_value)


    def test_setup_summary(self):
        print("Testing writting setup summary to tensorbaord")
        summary_placeholders, update_ops, summary_op = self.agent.setup_summary()
        print(summary_placeholders)
        print(update_ops)
        print(summary_op)


    def test_load_network(self):
        print("Testing loading the network")
        self.agent.load_network()


    def test_preprocess(self):
        print("Testing preprocessing states")
        stack = dqn.preprocess(np.array([100,100]),np.array([100,100]))
        print(stack)



if __name__ == '__main__':
    unittest.main()