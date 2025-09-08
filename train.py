from ENVtest import uavENV
import MADDPG



def train(env, agent, n_episodes, max_steps):
    env = uavENV(4)
    n_agents = env.n_agents
    
    
    all_matrix = {
        'episode_reward': [],
        'episode_step': [],
        'episode_energy': [],
        'episode_power': [],
        'episode_ddl': [],
        'episode_resource': [],
        'episode_energy_consumption': [],
    }

    for episode_i in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        for step_i in range(max_steps):
            pass