import numpy as np
import math



class uavENV:

    def __init__(self, n_uavs, n_ues):
        self.height = self.width = self.length =100 # m
        self.t_fly = 1.0 
        self.t_com = 7.0
        self.delta_t = self.t_fly + self.t_com

        self.alpha0 = 1e-5  # 参考信道增益
        self.p_noisy_los = 1e-13
        self.p_noisy_nlos = 1e-11
        self.B = 1e6  # per-bandwidth baseline (Hz). 可按需要扩展
        self.r = 10 **(-17)
        self.f_uav = 1.2e9  # UAV边缘服务器计算频率
        self.f_ue = 2e8
        self.s = 1000  # cycles per bit
        self.r_chip = 1e-27  # cpu power model coeff (UAV server)

        # mec_env 参数（reward 相关）
        self.LAMBDA_E = 0.5
        self.LAMBDA_T = 0.5
        
        self.CAPABILITY_E = 4  # GHz equivalent used in mec_env for server proc (we use f_uav for UAV server)
        self.N_UNITS = 8  # server parallel units (来自 mec_env)
        self.MAX_DDL = 1.0  # 最大允许时延（s） - adapt from mec_env
        # task size range (bits)
        self.MIN_TASK = int(1 * 1024 * 1024)  # ~1 Mbit in bits approx (user files used bytes->bits; 这里简化)
        self.MAX_TASK = int(4 * 1024 * 1024)


        self.n_uavs = n_uavs
        self.n_ues = n_ues
        self.M_uav = 10 # kg
        self.uav_velocity = 5.0 # m/s
        self.ue_velocity = 1.0 # m/s
        
        self.uav_locations = np.zeros((self.n_uavs, 2))
        self.uav_battery = np.zeros(self.n_uavs)

        self.ue_locations = np.zeros((self.n_ues, 2))
        self.ue_tasks = np.zeros(self.n_ues, dtype=np.float64)
        self.ue_block = np.zeros(self.n_ues, dtype=int)

        self.step_count = 0
        self.max_steps = 500

        self.state_dim = 3 + 3 * self.n_ues
        self.action_dim = 4
        self.action_bound = [-1.0, 1.0]

    def reset(self):
        
        for i in range(self.n_uavs):
            self.uav_locations[i] = np.random.uniform(0, self.width, size=2)
            self.uav_battery[i] = 50000.0

        
        self.ue_locations = np.random.uniform(0, self.width, size=(self.n_ues, 2))
        self.ue_tasks = np.random.randint(self.MIN_TASK, self.MAX_TASK + 1, size=self.n_ues).astype(np.float64)
        self.ue_block = np.random.randint(0, 2, size=self.n_ues)

        self.step_count = 0

        return self.get_obs_all()
    

    def get_obs_all(self):

        obs_list=[]
        ue_locs_flat = self.ue_locations.flatten()
        for i in range(self.n_uavs):
            obs = np.concatenate([
                np.array([self.uav_battery[i]]),
                self.uav_locations[i],
                ue_locs_flat,
                self.ue_tasks
            ])
            obs_list.append(obs)

        return np.array(obs_list)
        

    def channel_gain(self, uav_location, ue_location, block_flag):
        x= uav_location[0]-ue_location[0]
        y= uav_location[1]-ue_location[1]
        d= math.sqrt(x*x + y*y + self.height*self.height)
        g = abs(self.alpha0 / (d ** 2))
        p_noise = self.p_noisy_los if block_flag == 0 else self.p_noisy_nlos

        return g, p_noise, d
    
    def step(self, actions):

        rewards = np.zeros(self.n_uavs)
        task_completed = np.zeros(self.n_ues, dtype=bool)
        uav_energy = np.zeros(self.n_uavs)
        time = np.zeros(self.n_uavs)
        offload_records = []
        

        for i in range(self.n_uavs):
            action = np.clip((actions[i] + 1) / 2 , 0, 1)
            action = np.nan_to_num(action, nan=0.0)
            # print(f"action: {action}")
            ue_id = int(action[0] * self.n_ues) % self.n_ues
            theta = float(action[1] * 2 * np.pi)
            dist_scalar = action[2]
            fly_dist = dist_scalar * self.uav_velocity * self.t_fly
            offload_ratio = action[3]

            e_fly = 0.5 * self.M_uav * self.t_fly * (fly_dist / self.t_com)**2
            new_uav_location = self.update_uav_location(self.uav_locations[i], theta, fly_dist)

            ue_task_size = self.ue_tasks[ue_id]
            if ue_task_size <= 0:
                self.uav_battery[i] -= e_fly
                self.uav_locations[i] = new_uav_location
                rewards[i] = -1
                continue

            g, p_noise, d = self.channel_gain(new_uav_location, self.ue_locations[ue_id], int(self.ue_block[ue_id]))
            p_uplink = 0.1

            snr = max(0.0, p_uplink * g / p_noise)
            trans_rate = self.B * math.log2(1 + snr)

            bits_off = offload_ratio * ue_task_size
            bits_local = (1 - offload_ratio) * ue_task_size

            if trans_rate <= 0:
                t_tr = 1e6
            else:
                t_tr = bits_off / trans_rate
            e_trans = p_uplink * t_tr

            t_edge = bits_off * self.s / self.f_uav
            t_local = bits_local * self.s / self.f_ue

            e_edge = self.r * (self.f_uav **3) * t_edge * 10
            e_local = self.r * (self.f_ue **3) * t_local * 10

            offload_records.append({
                "uav": i,
                "ue": ue_id,
                "t_tr": t_tr,
                "t_edge": t_edge,
                "t_local": t_local,
                "bits_off": bits_off,
                "bits_local": bits_local,
                "new_uav_location": new_uav_location,
                "e_trans": e_trans,
                "e_edge": e_edge,
                "e_local": e_local,
                "e_fly": e_fly
            })

        if len(offload_records) > 0:
            sorted_idx = np.argsort([r["t_tr"] for r in offload_records])
            MECtime = np.zeros(self.N_UNITS)

            for rank in sorted_idx:
                record = offload_records[rank]
                uav = record["uav"]
                ue = record["ue"]

                if record["bits_off"] <= 0:
                    self.uav_locations[uav] = record["new_uav_location"]
                    self.uav_battery[uav] -= record["e_fly"]

                    time_norm = min(record["t_local"], self.MAX_DDL) / self.MAX_DDL

                    rewards[uav] += -(self.LAMBDA_E * record["e_fly"] + self.LAMBDA_T * time_norm)
                    continue

                if np.any(MECtime == 0):

                    finish = record["t_tr"] + record["t_edge"]
                else:
                    earliest = np.min(MECtime)
                    finish = max(record["t_tr"], earliest) + record["t_edge"]
                    
                MECtime[np.argmin(MECtime)] = finish
                time[uav] = finish

                time_off_norm = min(finish, self.MAX_DDL) / self.MAX_DDL

                time_local_norm = min(record["t_local"], self.MAX_DDL) / self.MAX_DDL

                uav_energy = record["e_trans"] + record["e_edge"] + record["e_fly"]
                rewards[uav] += -(self.LAMBDA_E * uav_energy + self.LAMBDA_T * (time_off_norm + time_local_norm))

                self.uav_battery[uav] -= uav_energy
                self.uav_locations[uav] = record["new_uav_location"]

                self.ue_tasks[ue] -= (record["bits_off"] + record["bits_local"])
                if self.ue_tasks[ue] <= 0:
                    task_completed[ue] = True
        else:
            pass
        
        self.update_ue()

        self.uav_battery = np.clip(self.uav_battery, 0.0, 1e9)

        for i in range(self.n_uavs):
            if self.uav_battery[i] <= 0:
                rewards[i] += -100.0

        self.step_count += 1
        done = (self.step_count >= self.max_steps) or (np.all(self.ue_tasks <= 0))

        info = {
            "task_completed": task_completed,
            "ue_tasks": self.ue_tasks,
            "uav_battery": self.uav_battery,
            "energy": uav_energy,
            "time": time,
            "offload_ratio": offload_ratio,
        }
                    
        return self.get_obs_all(), rewards, done, info

    def update_ue(self):
        for i in range(self.n_ues):
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * 2 * np.pi
            dist_ue = tmp[1] * self.delta_t * self.ue_velocity
            self.ue_locations[i,0] = np.clip(self.ue_locations[i,0] + dist_ue * math.cos(theta_ue), 0, self.width)
            self.ue_locations[i,1] = np.clip(self.ue_locations[i,1] + dist_ue * math.sin(theta_ue), 0, self.length)

            arrival = np.random.rand() < 0.1
            if arrival:
                self.ue_tasks[i] += np.random.randint(self.MIN_TASK//10, self.MIN_TASK//2)


    def update_uav_location(self, uav_location, theta, dist):
        x = np.clip(uav_location[0] + dist * math.cos(theta), 0, self.width)
        y = np.clip(uav_location[1] + dist * math.sin(theta), 0, self.length)
        return np.array([x, y])