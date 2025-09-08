import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Actor(nn.Module):


    def __init__(self, input_dim, action_dim, fc1_dims, fc2_dims, lr_actor):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = F.softmax(self.pi(x), dim=1)

        return mu
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
        
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, fc1_dims, fc2_dims, n_agents, lr_critic):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim * n_agents, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class Agent:

    def __init__(self, memo_size, obs_dim, state_dim, n_agent, action_dim, alpha
                 ,beta, fc1_dims, fc2_dims, gamma, tau , batch_size):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                           action_dim=action_dim).to(device)

        self.critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                             n_agent=n_agent,action_dim=action_dim).to(device)


        self.target_actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, 
                                  action_dim=action_dim).to(device)

        self.target_critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                             n_agent=n_agent, action_dim=action_dim).to(device)
        
        self.replay_buffer = ReplayBuffer(capacity=memo_size, obs_dim=obs_dim, state_dim=state_dim, 
                                          action_dim=action_dim, batch_size=batch_size)
        
    def get_action(self, obs):
        single_obs = torch.tensor(data=obs, dtype=torch.float).unsqueeze(0).to(device)
        single_action = self.actor.forward(single_obs)
        noise = torch.randn(self.action_dim).to(device) * 0.2
        single_action = torch.clamp(input=single_action + noise, min=0.0, max=1.0)

        return single_action.squeeze(0).detach().cpu().numpy()[0]
    
    def save_model(self, filename):
        self.actor.save_checkpoint(filename)
        self.target_actor.save_checkpoint(filename)
        self.critic.save_checkpoint(filename)
        self.target_critic.save_checkpoint(filename)

    def load_model(self, filename):
        self.actor.load_checkpoint(filename)
        self.target_actor.load_checkpoint(filename)
        self.critic.load_checkpoint(filename)
        self.target_critic.load_checkpoint(filename)
    




