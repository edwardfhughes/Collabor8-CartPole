import numpy as np

class ShoprEnv:

    def __init__(self):
        self.ideal_list = [0, 1, 1, 0, 1, 1, 0, 1]
        self.num_items = 8
        self.max_same_item = 1
        self.reset()

    def sample_action(self):
        return np.random.randint(0,self.num_items + 1)

    def step(self, action):
        done = False
        reward = -0.05

        if action == self.num_items:
            done = True
            for i in range(0, self.num_items):
                diff = self.state[i] - self.ideal_list[i]
                if diff > 0:
                    reward -= 0 # diff
                elif diff == 0:
                    reward += 1
                else:
                    reward += 0 #diff
        elif self.state[action] != self.max_same_item:
            self.state[action] += 1
        else:
            reward -= 0.5

        return self.state, reward, done, {}

    def reset(self):
        self.state = [0 for i in range(self.num_items)]
        return self.state
