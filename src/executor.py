import os
import time
from PIL import Image
import imageio

from src.environment import GymEnvironment


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
        timestamp = self._get_timestamp()
        self.game = "{}/game-{}".format(self.current_execution, timestamp)
        os.makedirs(self.game, exist_ok=True)

    def add_screenshots(self, screenshots):
        """
        Saves all game screenshots in a single folder
        """
        if not self.game:
            raise Exception("The method `new_game()` is not called")
        else:
            image_titles = []
            os.makedirs("{}/screenshots".format(self.game), exist_ok=True)
            for screenshot in screenshots:
                image = Image.fromarray(screenshot)
                timestamp = self._get_timestamp(milliseconds=True)
                image_title = "{}/screenshots/screen{}.png".format(self.game, timestamp)
                image.save(image_title)
                image_titles.append(image_title)
            self._generate_gif(image_titles)

    @staticmethod
    def _get_timestamp(milliseconds=False):
        if milliseconds:
            return time.time()
        else:
            return int(time.time())

    def _create_execution(self):
        timestamp = self._get_timestamp()
        self.current_execution = "{}/exec-{}".format(self.execution_path, timestamp)
        os.makedirs(self.current_execution, exist_ok=True)

    def _generate_gif(self, image_titles):
        images = []
        for filename in image_titles:
            images.append(imageio.imread(filename))
        imageio.mimsave("{}/game.gif".format(self.game), images)


def test_executor(ex, env_name):
    env = GymEnvironment(env_name)
    env.reset()
    screenshots = []
    ex.new_game()
    for _ in range(10):
        observation, reward, done, info = env.step(0)
        screenshots.append(observation)

    ex.add_screenshots(screenshots)


if __name__ == '__main__':
    executor = Executor("summary")
    test_executor(executor, 'Atlantis-v0')
