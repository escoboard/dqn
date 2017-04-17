import src.executor as executor
import src.dqn as dqn
import src.game_data as game_data
import src.graph as graph
import src.environment as environment
import numpy as np

import unittest


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.executor = executor.Executor()
        self.dqn = dqn.Agent()
        self.game_data = game_data.GameBatchData()
        self.graph = graph.Graph()
        self.environment = environment.GymEnvironment('Pong-vo')

    def test_integration_game_data_and_executor(self):
        print("Integration testing of game data and executor module")
        self.executor.new_game(11111)
        self.executor.add_screenshots(np.array([[100, 100], [100, 100]]))
        self.executor._generate_gif(np.array([[100, 100], [100, 100]]))
        self.executor.write_csv(['h1', 'h2'], [[1, 2], [2, 3]], [1, 2])
        print("Successful Integration testing of game data and executor module")


    def test_integration_environment_and_graph_execution(self):
        print("Integration testing of game data and executor module")
        self.environment.actions()
        self.environment.action_space()
        self.environment.make()
        self.environment.render()
        self.environment.step(0)
        s, q_values, model = self.agent.build_network()
        self.trainable_weights = model.trainable_weights
        a, y, loss, grads_update = self.agent.build_training_op(self.trainable_weights)
        stack = self.agent.get_initial_state(np.array([[100, 100], [100, 100]]))
        action, action_value = self.agent.get_action(np.array([[100, 100], [100, 100]]))
        summary_placeholders, update_ops, summary_op = self.agent.setup_summary()
        self.agent.load_network()
        stack = dqn.preprocess(np.array([100,100]),np.array([[100, 100], [100, 100]]))
        print(s)
        print(q_values)
        print(model)
        print(a)
        print(y)
        print(loss)
        print(grads_update)
        print(stack)
        print(action)
        print(action_value)
        print(summary_placeholders)
        print(update_ops)
        print(summary_op)
        print(stack)
        print("Successful Integration testing of game data and executor module")


if __name__ == '__main__':
    unittest.main()