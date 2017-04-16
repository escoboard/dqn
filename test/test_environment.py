from src.environment import GymEnvironment
import unittest
class TestGymEnvironment(unittest.TestCase):
    def __init__(self):
        self.environment = GymEnvironment('Pong-vo')

    def test_number_of_action_pong(self):
        print("Testing number of action given by Pong-vo environment")
        NUMBER_OF_ACTION_PONGV0 = 6
        self.assertEqual(NUMBER_OF_ACTION_PONGV0, self.environment.actions())




if __name__ == '__main__':
    unittest.main()