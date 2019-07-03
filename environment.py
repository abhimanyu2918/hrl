import random
import numpy as np

class StochasticMDP(object):
    """Environment for the agent."""
    def __init__(self):
        super(StochasticMDP, self).__init__()
        self.start_state = 2
        self.num_states = 6
        self.num_actions = 2
        self.p_right = 0.5
        self.end = False
        self.current_state = self.start_state

    def reset(self):
        self.end = False
        self.current_state = self.start_state
        state = np.zeros(self.num_states)
        state[self.current_state-1] = 1
        return state

    def step(self, action):
        #deciding the next state based on chosen action
        if(self.current_state != 1):
            if action == 2:
                #action is to move right
                if random.random() < self.p_right:
                    if self.current_state != self.num_states:
                        self.current_state += 1
                else:
                    self.current_state -= 1
            else:
                #action is to move left
                self.current_state -= 1

            if self.current_state == self.num_states:
                self.end = True

        state = np.zeros(self.num_states)
        state[self.current_state-1] = 1

        #deciding reward
        if self.current_state == 1:
            if self.end:
                return state, 1.00, True, {}
            else:
                return state, 1.00/100.00, True, {}
        else:
            return state, 0.0, False, {}
