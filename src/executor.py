import os
import time


class Executor:
    def __init__(self, execution_path):
        """
        New Execution of the game
        Creates a new folder for the current execution
        """
        self.execution_path = execution_path
        self.game = None
        self.current_execution = None
        self._create_execution()

    def new_game(self):
        """
        Creates a folder 'game-TIMESTAMP'
        """
        timestamp = int(time.time())
        self.game = "{}/game-{}".format(self.current_execution, timestamp)
        os.makedirs(self.game, exist_ok=True)

    def add_screenshots(self, screenshots):
        """
        Saves all game screenshots in a single folder
        """
        pass

    def _create_execution(self):
        timestamp = int(time.time())
        self.current_execution = "{}/exec-{}".format(self.execution_path, timestamp)
        os.makedirs(self.current_execution, exist_ok=True)

    def _generate_gif(self):
        pass


if __name__ == '__main__':
    executor = Executor("summary")
    for _ in range(5):
        executor.new_game()
        time.sleep(1)