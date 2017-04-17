from src.executor import Executor
import unittest
import numpy as np

class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = Executor()

    def test_add_new_game(self):
        print("Testing creation of new game")
        self.executor.new_game(timestamp=111111111)

    def test_add_screenshots(self):
        print("Testing creation of new execution")
        self.executor.add_screenshots(np.array([[100, 100], [100, 100]]))


    def test_write_csv(self):
        print("Testing writting to csv")
        self.executor.write_csv(['h1','h2'],[[1,2],[2,3]],[1,2])

    def test_create_execution(self):
        print("Testing creation of new execution")
        self.executor._create_execution(timestamp=111111111)


    def test_generate_gif(self):
        print("Testing generation of gif")
        self.executor._generate_gif(np.array([[100, 100], [100, 100]]))


if __name__ == '__main__':
    unittest.main()

