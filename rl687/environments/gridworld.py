import numpy as np
from .skeleton import Environment

class Gridworld(Environment):
    """
    The Gridworld as described in the lecture notes of the 687 course material. 
    
    Actions: up (0), right (1), down (2), left (3),
    
    Environment Dynamics: With probability 0.8 the robot moves in the specified
        direction. With probability 0.05 it gets confused and veers to the
        right -- it moves +90 degrees from where it attempted to move, e.g., 
        with probability 0.05, moving up will result in the robot moving right.
        With probability 0.05 it gets confused and veers to the left -- moves
        -90 degrees from where it attempted to move, e.g., with probability 
        0.05, moving right will result in the robot moving down. With 
        probability 0.1 the robot temporarily breaks and does not move at all. 
        If the movement defined by these dynamics would cause the agent to 
        exit the grid (e.g., move up when next to the top wall), then the
        agent does not move. The robot starts in the top left corner, and the 
        process ends in the bottom right corner.
        
    Rewards: -10 for entering the state with water
            +10 for entering the goal state
            0 everywhere else
        
    
    
    """

    def __init__(self, startState=0, endState=24, shape=(5,5), obstacles=[12, 17], waterStates=[6, 18, 22]):
        self.startState = startState
        self.endState = endState
        self.shape = shape
        self.obstacles = obstacles
        self.waterStates = waterStates

        self._state = startState
        self._action = -1    # no action taken yet
        self._reward = 0     # no reward yet
        self._gamma = 0.9
        self.gamma_coeff = 1
        
    @property
    def name(self):
        pass

    @property
    def reward(self):
        return self._reward

    @property
    def action(self):
        return self._action

    @property
    def isEnd(self):
        return self._state == self.endState

    @property
    def state(self):
        return self._state

    @property
    def gamma(self):
        return self._gamma

    def step(self, action):
        p = np.random.random_sample()
        # veer right with prob 0.05
        actual_action = action
        if p < 0.05:
            actual_action = (action+1)%4
        # veer left with prob 0.05
        elif p < 0.1:
            actual_action = (action-1)%4
        # stay there with prob 0.1
        elif p < 0.2:
            actual_action = -1

        # move up if possible
        if (actual_action == 0):
            new_state = self._state - self.shape[1]
            if (new_state >= 0 and new_state not in self.obstacles):
                self._state = new_state
        # move right if possible
        elif (actual_action == 1):
            new_state = self._state + 1
            if (self._state%5 != 4 and new_state not in self.obstacles):
                self._state = new_state
        # move down if possible
        elif (actual_action == 2):
            new_state = self._state + self.shape[1]
            if (new_state < 25 and new_state not in self.obstacles):
                self._state = new_state
        # move left if possible
        elif (actual_action == 3):
            new_state = self._state - 1
            if (self._state%5 != 0 and new_state not in self.obstacles):
                self._state = new_state

        self._action = action           # Update A_t
        self._reward = self.gamma_coeff * self.R(self._state)  # Update R_t
        self.gamma_coeff *= self._gamma
        return (actual_action, self._state, self._reward)


    def reset(self):
        self._state = self.startState
        self._action = -1    # no action taken yet
        self._reward = 0     # no reward yet
        self._gamma = 0.9
        self.gamma_coeff = 1
        
    def R(self, _state):
        """
        reward function
        
        output:
            reward -- the reward resulting in the agent being in a particular state
        """
        if _state == self.endState:
            self._reward = 10
        elif _state in self.waterStates:
            self._reward = -10
        else:
            self._reward = 0
        return self._reward
