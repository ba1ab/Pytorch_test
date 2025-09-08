import numpy as np
from copy import deepcopy

LAMBDA_E = 0.5
LAMBDA_T = 0.5
MIN_SIZE = 1 # MB  1024 * 8 
MAX_SIZE = 50 # MB 1024 * 8
MIN_CYCLE = 300 # 
MAX_CYCLE = 1000
MIN_DDL = 0.1 # seconds
MAX_DDL = 1 # seconds
MIN_RESOURCE = 0.4 # GHz
MAX_RESOURCE = 1.5 # GHz
MIN_POWER = 1 # dB
MAX_POWER = 24
CAPABILITY_E = 4 # GHz 
K_ENERGY_LOCAL = 5 * 1e-27

MIN_ENE =0.5
MAX_ENE = 3.2
HARVEST_RATE = 0.001
W_BANDWIDTH = 40

K_CHANNEL = 10
S_E = 400
N_UNITS = 8
MAX_STEPS = 10



class uavENV(object):

    

    def __init__(self, n_agents):
        
        self.state_size = 7
        self.action_size = 3
        self.n_agents = n_agents
        self.W_BANDWIDTH = W_BANDWIDTH

        self.S_power = np.zeros(self.n_agents)
        self.initial_energy = np.zeros(self.n_agents)
        self.S_energy = np.zeros(self.n_agents)
        self.S_gain = np.zeros(self.n_agents)
        self.S_size = np.zeros(self.n_agents)
        self.S_cycle = np.zeros(self.n_agents)  
        self.S_ddl = np.zeros(self.n_agents)
        self.S_res = np.zeros(self.n_agents)
        self.action_lower_bound = [0,  0.01, 0.01] 
        self.action_higher_bound = [1, 1, 1]
        for n in range(self.n_agents):
            self.S_size[n] = np.random.uniform(MIN_SIZE, MAX_SIZE)
            self.S_cycle[n] = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
            self.S_ddl[n] = np.random.uniform(MIN_DDL, MAX_DDL)
            self.S_res[n] = np.random.uniform(MIN_RESOURCE, MAX_RESOURCE)

    
    
    def reset(self):
        self.step = 0
        for n in range(self.n_agents):
            self.S_size[n] = np.random.uniform(MIN_SIZE, MAX_SIZE)
            self.S_cycle[n] = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
            self.S_ddl[n] = np.random.uniform(MIN_DDL, MAX_DDL)
            self.S_energy[n] = deepcopy(self.initial_energy[n])
        self.S_energy = np.clip(self.S_energy, MIN_ENE, MAX_ENE)
        self.state = np.array([[self.S_power[n], self.S_gain[n], self.S_energy[n], self.S_size[n], self.S_cycle[n], self.S_ddl[n], self.S_res[n]] for n in range(self.n_agents)])
        return self.state
    
    
    
    def step_mec(self, action):
        A_decision = np.zeros(self.n_agents)
        A_res = np.zeros(self.n_agents)
        A_power = np.zeros(self.n_agents)
        for n in range(self.n_agents):
            A_decision[n] = action[n][0]
            A_res[n] = action[n][1] * self.S_res[n] * 10 ** 9
            A_power[n] = action[n][2] * 10 ** ((self.S_power[n]-30)/10)

        # 任务时间计算
        x_n = A_decision
        DataRate = self.W_BANDWIDTH * 10 ** 6  * np.log(1 + A_power * 10 **(self.S_gain/10)) / np.log(2)
        DataRate = DataRate / K_CHANNEL
        Time_proc = self.S_size*8*1024*self.S_cycle / (CAPABILITY_E*10**9)
        Time_local = self.S_size*8*1024*self.S_cycle / A_res
        Time_max_local = self.S_size*8*1024*self.S_cycle / (MIN_RESOURCE*10**9)
        Time_off = self.S_size*8*1024/ DataRate
        for i in range(self.n_agents):
            if x_n[i] == 2:
                Time_off[i] = MAX_DDL
                x_n[i] = 1
        Time_finish = np.zeros(self.n_agents)
        SortedOFF = np.argsort(Time_off)
        MECtime = np.zeros(N_UNITS)
        counting = 0
        for i in range(self.n_agents):
            if x_n[SortedOFF[i]] == 1 and counting < N_UNITS:
                Time_finish[SortedOFF[i]] = Time_local[SortedOFF[i]] + Time_proc[SortedOFF[i]]
                MECtime[np.argmin(MECtime)] = Time_local[SortedOFF[i]] + Time_proc[SortedOFF[i]]
                counting += 1
            elif x_n[SortedOFF[i]] == 1:
                for j in range(i):
                    if x_n[SortedOFF[j]] == 1:
                        MECtime[np.argmin(MECtime)] += Time_proc[SortedOFF[j]]
                Time_finish[SortedOFF[i]] = max(Time_off[SortedOFF[i]], np.min(MECtime)) + Time_proc[SortedOFF[i]]
                MECtime[np.argmin(MECtime)] = max(Time_off[SortedOFF[i]], np.min(MECtime)) + Time_proc[SortedOFF[i]]
        Time_n = (1-x_n) * Time_local + x_n * (Time_off + Time_proc)

        Time_n = [min(t,MAX_DDL) / MAX_DDL for t in Time_n]
        T_mean = np.mean(Time_n)

        # 能耗计算
        Energy_local = K_ENERGY_LOCAL * self.S_size *8*1024 * self.S_cycle * A_res
        Energy_max_local = K_ENERGY_LOCAL * self.S_size *8*1024* self.S_cycle * (self.S_res*10**9)
        Energy_off = A_power * Time_off
        Energy_n = (1-x_n) * Energy_local + x_n * Energy_off
        self.S_energy = np.clip(self.S_energy - Energy_n*1e-6 + np.random.normal(HARVEST_RATE, 0, size=self.n_agents)*1e-6, 0, MAX_ENE)
        for i in range(x_n.size):
            if self.S_energy[i] <= 0:
                Time_n[i] = MAX_DDL / MIN_DDL
        
        # 奖励计算
        
        Time_penalty = np.maximum((Time_n - self.S_ddl/MAX_DDL), 0)
        Energy_penalty = np.maximum((MIN_ENE - self.S_energy), 0)*10**6
        time_penalty_nozero_count = np.count_nonzero(Time_penalty)/self.n_agents
        energy_penalty_nozero_count = np.count_nonzero(Energy_penalty)/self.n_agents
        Reward = -1*(LAMBDA_E * np.array(Energy_n) + LAMBDA_T * np.array(Time_n)) -1*(LAMBDA_E *np.array(Energy_penalty) + LAMBDA_T*np.array(Time_penalty))
        Reward = np.ones_like(Reward) * np.sum(Reward)
        for n in range(self.n_agents):
            self.S_size[n] = np.random.uniform(MIN_SIZE, MAX_SIZE)
            self.S_cycle[n] = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
            self.S_ddl[n] = np.random.uniform(MIN_DDL, MAX_DDL - MIN_DDL/10)
        
        # 状态更新
        self.state = np.array([[self.S_power[n], self.S_gain[n], self.S_energy[n], self.S_size[n], self.S_cycle[n], self.S_ddl[n], self.S_res[n]] for n in range(self.n_agents)])
        self.step += 1
        done = False
        if self.step >= MAX_STEPS:
            self.step = 0
            done = True
        return self.state, Reward, done, energy_penalty_nozero_count, time_penalty_nozero_count