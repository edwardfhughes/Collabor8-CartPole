import numpy as np


class ShoprEnv:

    def __init__(self):
        self.ideal_list = [1, 1, 1, 1]
        self.num_iterations = sum(self.ideal_list)
        self.num_items = 4
        self.reset()

    def sample_action(self):
        return np.random.randint(0,self.num_items + 1)

    def step(self, action):
        self.current_iteration += 1
        done = False
        reward = 0

        if self.current_iteration > self.num_iterations + 1:
            reward = 0
            done = True
            return self.state, reward, done, {}

        if action != self.num_items:
            self.state[action] += 1

        if self.current_iteration == self.num_iterations + 1:
            for i in range(self.num_items):
                if self.ideal_list[i] == self.state[i]:
                    reward += 10
                else:
                    reward -= 0
            return self.state, reward, done, {}
        return self.state, reward, done, {}

        # if self.state[action] <= self.ideal_list[action]:
        #     reward += 1 / self.ideal_list[action]

        # if self.state[action] > self.ideal_list[action]:
        #    reward -= 1 / (self.ideal_list[action] + 1)

    def reset(self):
        self.state = [0 for i in range(self.num_items)]
        self.current_iteration = 0
        return self.state

class BuzzEnv:

    def __init__(self):
        self.num_iterations = 4
        self.num_items = 4
        self.reset()

    def sample_action(self):
        return np.random.randint(0,self.num_items + 1)

    def step(self, action):
        self.current_iteration += 1
        done = False
        reward = 0

        if self.current_iteration > self.num_iterations:
            done = True
            return self.state, reward, done, {}

        if action == self.num_items:
            return self.state, reward, done, {}

        if action == 0 and sum(self.state) == 0:
            reward += 1
        elif self.state[action] == 0 and self.state[action - 1] == 1:
            reward += 1

        self.state[action] += 1

        return self.state, reward, done, {}

        # if self.state[action] <= self.ideal_list[action]:
        #     reward += 1 / self.ideal_list[action]

        # if self.state[action] > self.ideal_list[action]:
        #    reward -= 1 / (self.ideal_list[action] + 1)

    def reset(self):
        self.state = [0 for i in range(self.num_items)]
        self.current_iteration = 0
        return self.state
