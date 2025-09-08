import numpy as np


class uavENV:

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.state_dim = 4
        self.action_dim = 2