import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from ENV import uavENV
from MADDPG import MADDPG_Agent
from ReplayBuffer import ReplayBuffer
import matplotlib.pyplot as plt
import pickle

def pad_state(state, target_dim):
    return np.pad(state, (0, target_dim - len(state)), 'constant')

def train_agents(env, n_episodes = 10):
    # print(f"state dim: {env.state_dim}")
    n_uavs = env.n_uavs
    s_dim = env.state_dim
    a_dim = env.action_dim

    all_metrics = {
        'episode_rewards': {},
        'episode_offloading_ratios': {},
        'episode_Time_': {},
        'episode_energy_': {}
    }
    
    for i in range(1):
        lr_acotr = 1e-4
        lr_critic = 1e-3
        agents = [MADDPG_Agent(s_dim, a_dim, env.action_bound, lr_acotr, lr_critic) for _ in range(n_uavs)]
        replay_buffers = [ReplayBuffer(200000, state_dim = s_dim, action_dim = a_dim) for _ in range(n_uavs)]

        episode_rewards = []
        episode_offloading_ratios = []
        episode_Time_ = []
        episode_energy_ = []

        for episode in range(n_episodes):
            print(f"Episode: {episode + 1}")
            state = env.reset()
            # print(f"State shape after reset: {state.shape}")
            print(f"state: {state}")
            episode_reward = 0
            offloading_ratios = []
            energy_ = []
            Time_ = []


            while True:
                actions = []
                for j in range(n_uavs):
                    #print(f"j: {j}")
                    uav_state = state[j]
                    # print(uav_state)
                    # print(f"State shape before padding: {uav_state.shape}")
                    # uav_state = pad_state(uav_state, s_dim)
                    # print(f"State shape after padding: {uav_state.shape}")
                    action = agents[j].choose_action(uav_state)
                    actions.append(action)

                next_state, rewards, done, info = env.step(actions)
                

                offloading_ratios_step = info["offload_ratio"]
                energy_step = info["energy"]
                Time_step = info["time"]

                for j in range(n_uavs):
                    # uav_state = pad_state(state[j * s_dim:(j + 1) * s_dim], s_dim)
                    
                    # uav_next_state = pad_state(next_state[j * s_dim:(j + 1) * s_dim], s_dim)
                    
                    replay_buffers[j].store(state=state[j], action=actions[j], reward=rewards[j], next_state=next_state[j], done=done)
                    agents[j].learn(replay_buffers[j], batch_size=128)  # Larger batch size

                state = next_state
                episode_reward += np.nansum(rewards)

                offloading_ratios.append(np.mean(offloading_ratios_step))
                
                Time_.append(np.mean(Time_step))
                energy_.append(np.mean(energy_step))

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_offloading_ratios.append(np.mean(offloading_ratios))

            episode_Time_.append(np.mean(Time_))
            episode_energy_.append(np.mean(energy_))
            print(f"Total Reward for Episode {episode + 1}: {episode_reward}")

        lr_key = f'10^-{i}'
        all_metrics['episode_rewards'][lr_key] = episode_rewards
        all_metrics['episode_offloading_ratios'][lr_key] = episode_offloading_ratios

        all_metrics['episode_Time_'][lr_key] = episode_Time_
        all_metrics['episode_energy_'][lr_key] = episode_energy_

    return all_metrics

# Plotting functions (unchanged)
def plot_all_learning_rates(all_metrics, metric_name, ylabel):
    plt.figure(figsize=(12, 6))
    for lr, values in all_metrics[metric_name].items():
        plt.plot(values, label=f'LR: {lr}')
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(f'{metric_name.replace("_", " ").title()} vs Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_name}_all_lr.png')
    plt.show()
    plt.close()

# Main execution
if __name__ == "__main__":
    env = uavENV(n_uavs=4, n_ues=2)  # Assuming UAVEnv is defined elsewhere
    all_metrics = train_agents(env)

    with open('DDPG', 'wb') as file:
        pickle.dump(all_metrics, file)

    print('All metrics saved successfully')

    plot_all_learning_rates(all_metrics, 'episode_rewards', 'Total Reward')
    plot_all_learning_rates(all_metrics, 'episode_offloading_ratios', 'Average Offloading Ratio')
    plot_all_learning_rates(all_metrics, 'episode_Time_', 'Average Time')
    plot_all_learning_rates(all_metrics, 'episode_energy_', 'Average Energy')
