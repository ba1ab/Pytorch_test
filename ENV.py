import numpy as np




class uavENV:

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.height = self.width = self.length =100 # m

        self.uav_location = np.array([[0, 0] for _ in range(self.n_agents)])
        self.uav_velocity = 5 # m/s
        
        self.uav_energy = np.zeros(self.n_agents)
        self.uav_size = np.zeros(self.n_agents)



        self.n_ue = 4
        self.ue_location = np.array([[0, 0] for _ in range(self.n_ue)])
        self.ue_velocity = 2 # m/s

        for n in range(self.n_agents):
            
        

