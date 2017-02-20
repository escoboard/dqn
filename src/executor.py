import os
import csv
import imageio
from PIL import Image

from src.environment import GymEnvironment


class Executor:
    def __init__(self, execution_path, execution_timestamp=None):
        """
        New Execution of the game
        Creates a new folder for the current execution
        """
        self.execution_path = execution_path
        self.game = None
        self.current_execution = None
        self._create_execution(execution_timestamp)

    def new_game(self, timestamp=None):
        """
        Creates a folder 'game-TIMESTAMP'
        """
        self.game = "{}/game-{}".format(self.current_execution, timestamp)
        os.makedirs(self.game, exist_ok=True)

    def add_screenshots(self, screenshots, title='screenshots'):
        """
        Saves all game screenshots in a single folder
        """
        if not self.game:
            raise Exception("The method `new_game()` is not called")
        else:
            image_titles = []
            os.makedirs("{}/{}".format(self.game, title), exist_ok=True)
            for screenshot in screenshots:
                image = Image.fromarray(screenshot[0])
                timestamp = screenshot[1]
                image_title = "{}/{}/screen{}.png".format(self.game, title, timestamp)
                image.save(image_title)
                image_titles.append(image_title)
            if title == 'screenshots':
                self._generate_gif(image_titles)

    def write_csv(self, headers, data, game_timestamp):
        if not self.game:
            raise Exception("The method `new_game()` is not called")
        else:
            with open("{}/game-{}/data.csv".format(self.current_execution, game_timestamp), 'w') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(headers)
                for i, row in enumerate(data):
                    write_row_data = [i, row['timestamp'], row['reward'], row['action_value'], row['action'],
                                      row['loss']]
                    csv_writer.writerow(write_row_data)

    def _create_execution(self, timestamp=None):
        self.current_execution = "{}/exec-{}".format(self.execution_path, timestamp)
        os.makedirs(self.current_execution, exist_ok=True)

    def _generate_gif(self, image_titles):
        images = []
        for filename in image_titles:
            images.append(imageio.imread(filename))
        imageio.mimsave("{}/game.gif".format(self.game), images)
