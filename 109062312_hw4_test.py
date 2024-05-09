import gym
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from torch.distributions import Normal
import pdb
from collections import deque
from osim.env import L2M2019Env

class ActorNet(nn.Module):
    def __init__(self, max_action, input_dim=339, action_dim=22):
        super(ActorNet, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.max_action * torch.tanh(self.mean(x))
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std

class Agent:
    def __init__(self):
        self.policy = ActorNet(max_action=1.0)
        self.policy.load_state_dict(torch.load('109062312_hw4_data'))
        # self.policy = torch.load('109062312_hw4_data').to('cpu')
        # torch.save(self.policy.state_dict(), 'model_state_dict.pth')
        # import pdb
        # pdb.set_trace()
        self.min_action = 0.0
        self.max_action = 1.0

    def act(self, obs):
        state = self.preprocess_observation(obs)
        mean, std = self.policy(state.unsqueeze(0))
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, self.min_action, self.max_action).detach().cpu().numpy().squeeze()
        return action

    def preprocess_observation(self, obs):
        preprocessed_obs = np.empty(339)

        preprocessed_obs[:242] = obs['v_tgt_field'].flatten()

        pelvis_data = [obs['pelvis']['height'], obs['pelvis']['pitch'], obs['pelvis']['roll']] + obs['pelvis']['vel']
        preprocessed_obs[242:251] = pelvis_data

        def extract_leg_data(leg_data, start_index):
            leg_features = []
            leg_features.extend(leg_data['ground_reaction_forces'])
            leg_features.extend([leg_data['joint']['hip_abd'], leg_data['joint']['hip'],
                                leg_data['joint']['knee'], leg_data['joint']['ankle']])
            leg_features.extend([leg_data['d_joint']['hip_abd'], leg_data['d_joint']['hip'],
                                leg_data['d_joint']['knee'], leg_data['d_joint']['ankle']])
            for muscle in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                muscle_data = leg_data[muscle]
                leg_features.extend([muscle_data['f'], muscle_data['l'], muscle_data['v']])
            preprocessed_obs[start_index:start_index + 44] = leg_features

        extract_leg_data(obs['r_leg'], 251)
        extract_leg_data(obs['l_leg'], 295)

        return torch.from_numpy(preprocessed_obs).float()