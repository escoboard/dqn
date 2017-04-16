from src.executor import Executor
import unittest

class TestExecutor(unittest.TestCase):
    def __init__(self):
        self.executor = Executor()

    def test_add_new_game(self):
        print("Testing creation of new game")
        self.executor.new_game(timestamp=111111111)

    def test_add_screenshots(self):
        print("Testing creation of new execution")



    def test_write_csv(self):
        print("Testing writting to csv")

    def test_create_execution(self):
        print("Testing creation of new execution")
        self.executor._create_execution(timestamp=111111111)

    def test_generate_gif(self):
        print("Testing generation of gif")



if __name__ == '__main__':
    unittest.main()

