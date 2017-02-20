class GameBatchData:
    def __init__(self):
        self.data = []
        self.game_timestamps = []
        self.current_game_index = None
        self.steps = 0

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

    def get_total_steps(self):
        return self.steps