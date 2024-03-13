import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64):
        # QNetwork consists of three fully connected layer
        # state_size x fc1_size
        # fc1_size x fc2_size
        # fc2_size x action_size
        # input:
        # - state_size: number of observations
        # - action_size: number of possible actions
        # - seed: for reproducability
        # - fc1_size=64: number of firt inner layer
        # - fc2_size=64: number of second inner layer

        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        # forward pass to transform state to action extimation
        # input:
        # - state: observations
        # output:
        # - x: q value estimation

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
