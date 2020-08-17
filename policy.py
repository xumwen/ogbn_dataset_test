import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

CLIP_EPS = 0.1
sigma = 1

nb_episodes = 5
nb_epoches = 5


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
    
    def clear_mem(self):
        del self.states
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]


class Policy(nn.Module):
    def __init__(self, state_size, out_channels):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.out_channels = out_channels
        self.build_actor()

    def build_actor(self):
        self.actor = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU()
        )
        self.action_mean = nn.Linear(32, self.out_channels)
    
    def forward(self, state):
        act_hid = self.actor(state)
        action_mean = self.action_mean(act_hid)
        action_std = sigma

        # sample from n Normal and calculate sum of log_prob
        actions = torch.zeros(self.out_channels)
        log_prob = 0.0
        for i in range(self.out_channels):
            normal = Normal(action_mean[i], action_std)
            action = normal.sample()

            log_prob += normal.log_prob(action)
            actions[i] = action

        return actions, log_prob.item()

    def evaluate(self, state, action):
        act_hid = self.actor(state)
        action_mean = self.action_mean(act_hid)
        action_std = sigma

        log_prob = 0.0
        for i in range(self.out_channels):
            normal = Normal(action_mean[i], action_std)
            log_prob += normal.log_prob(action[i])

            # entropy += normal.entropy()

        return log_prob.item()


class PPO:
    def __init__(self, state_size, out_channels, device):
        self.policy = Policy(state_size, out_channels)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

        self.memory = Memory()
        self.device = device

    def make_batch(self):
        s = torch.stack(self.memory.states).to(self.device)
        a = torch.stack(self.memory.actions).to(self.device)
        logp = torch.FloatTensor(self.memory.logprobs).to(self.device).unsqueeze(dim=1)
        r = torch.FloatTensor(self.memory.rewards).to(self.device).unsqueeze(dim=1)

        return s, a, logp, r

    def train(self):
        self.policy.to(self.device)

        s, a, logp, r = self.make_batch()
        r_avg = r.mean()

        for i in range(nb_epoches):
            a_logprob = self.policy.evaluate(s, a)
            ratio = torch.exp(a_logprob - logp)

            advantage = r - r_avg

            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio,
                1 - CLIP_EPS,
                1 + CLIP_EPS
            ) * advantage

            loss = -torch.min(surr1, surr2).mean()

            print('Loss: {:.2f}'.format(loss.item()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.policy.cpu()

    def train_step(self, env):
        s = env.get_state()

        for eid in range(nb_episodes):
            a, logp = self.policy(s)
            r = env.step(a, eid)

            self.memory.states.append(s)
            self.memory.actions.append(a)
            self.memory.logprobs.append(logp)
            self.memory.rewards.append(r)

        self.train()
        self.memory.clear_mem()