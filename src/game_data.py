from executor import Executor


class GameBatchData:
    def __init__(self, timestamp):
        self.execution_timestamp = timestamp
        self.data = []
        self.game_timestamps = []
        self.current_game_index = None
        self.steps = 0
        self.game_data_headers = [
            'index', 'timestamp', 'reward',
            'action_value', 'action', 'loss'
        ]

    def new_game(self, timestamp):
        self.game_timestamps.append(timestamp)
        self.current_game_index = len(self.game_timestamps) - 1
        self.data.append([])
        return self.current_game_index

    def add_step(
            self, timestamp, observation,
            concatenated_observation, reward,
            action_value, action):
        step_observation = {
            'timestamp': timestamp,
            'observation': observation,
            'concatenated_observation': concatenated_observation,
            'reward': reward,
            'action_value': action_value,
            'action': action,
            'loss': None
        }
        self.steps += 1
        self.data[self.current_game_index].append(step_observation)
        data_index = len(self.data[self.current_game_index]) - 1
        return data_index

    def get_data(self):
        return self.data

    def save_progress(self):
        executor = Executor(execution_path="summary", execution_timestamp=self.execution_timestamp)
        for i, game in enumerate(self.data):
            game_timestamp = self.game_timestamps[i]
            executor.new_game(game_timestamp)

            # Save screenshots
            screenshots = []
            concatenated_screenshots = []
            for step in game:
                screenshots.append((step['observation'], step['timestamp']))
                #concatenated_screenshots.append((step['concatenated_observation'], step['timestamp']))
            executor.add_screenshots(screenshots)
            #executor.add_screenshots(concatenated_screenshots, 'concat_screenshots')

            # Write data to CSV
            executor.write_csv(headers=self.game_data_headers,
                               data=game, game_timestamp=game_timestamp)

    def get_total_steps(self):
        return self.steps
