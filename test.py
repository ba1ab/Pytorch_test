import torch
from MADDPG import MADDPG_Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = MADDPG_Agent(state_dim=9, action_dim=4, action_bound=[-1,1], lr_actor=1e-4, lr_critic=1e-3)

s = torch.randn(9)           # 单样本 state
a = agent.choose_action(s.numpy())
print("choose_action single:", a.shape)

# batch test
s_b = torch.randn(16, 9).numpy()
with torch.no_grad():
    a_b = agent.actor(torch.FloatTensor(s_b).to(agent.actor.fc1.weight.device))
print("actor batch out:", a_b.shape)

# critic forward single
from Model import Critic
critic = agent.critic
q = critic(torch.FloatTensor(s).unsqueeze(0).to(device), torch.FloatTensor(a).unsqueeze(0).to(device))
print("critic q single shape:", q.shape)

# critic forward batch
q_b = critic(torch.FloatTensor(s_b).to(device), a_b)
print("critic q batch shape:", q_b.shape)
