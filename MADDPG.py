import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model import Actor, Critic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPG_Agent:

    def __init__(self, state_dim, action_dim, action_bound, lr_actor, lr_critic):
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.target_actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-4)

        self.gamma = 0.99
        self.tau = 0.01


    def choose_action(self, state,):
        # print(f"state shape: {state.shape}")
        state = torch.FloatTensor(state).unsqueeze(0)
        # print(f"state tensor shape: {state.shape}")
        with torch.no_grad():
            action = self.actor(state).numpy().flatten()
        return action
    
    def update_target_network(self):
        tau = tau if tau is not None else self.tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def learn(self, replay_buffer, batch_size):
        if len(replay_buffer.buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        target_q = self.target_critic(next_state, self.target_actor(next_state))
        target_q = reward + (1 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_network()


