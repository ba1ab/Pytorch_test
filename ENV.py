import numpy as np




class uavENV:

    def __init__(self, n_uavs, n_ues):
        self.n_uavs = n_uavs
        self.n_ues = n_ues

        self.height = self.width = self.length =100 # m

        self.uav_velocity = 5.0 # m/s
        self.t_fly = 1.0 
        self.t_com = 7.0
        self.delta_t = self.t_fly + self.t_com

        self.uav_locations = np.zeros((self.n_uavs, 2))
        self.uav_battery = np.zeros(self.n_uavs)

        self.ue_locations = np.zeros((self.n_ues, 2))
        self.ue_tasks = np.zeros(self.n_ues, dtype=np.float64)

        self.step_coounter = 0


    def reset(self):
        

        
        

