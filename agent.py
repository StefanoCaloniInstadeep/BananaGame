import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

from network import QNetwork
from replay_buffer import ReplayBuffer


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64*4
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4


class Agent():
    def __init__(self, state_size, action_size, seed, device=torch.device("cpu")):
        # Agent is the key piece of the algorithm.
        # It stores 2 networks, one local and one target,
        # - qnetwork_local: the local network we optimize
        # - qnetwork_target: the target network that gets updated every UPDATE_EVERY step
        # - optimizer: the torch optimizer to train the local network
        # - memory: the replay buffer to random sampling previous experience
        # input:
        # - state_size: number of observations
        # - action_size: number of possible actions
        # - seed: for reproducability
        # - device: device to be used

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device

        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)

        self.t_step = 0

    def __repr__(self):
        return "Deep Qnetwork agent\n"+self.qnetwork_local.__repr__()

    def save_local_network(self, path):
        # save the checkpoint of local network
        # input:
        # - path: path of the file

        torch.save(self.qnetwork_local.state_dict(), path)

    def load_local_network(self, path):
        # load the checkpoint of local network
        # input:
        # - path: path of the file

        self.qnetwork_local.load_state_dict(torch.load(path))

    def step(self, state, action, reward, next_state, done):
        # save the current step to the memory
        # if hit the UPDATE_EVERY and we have enough memory, we train a bit
        # input:
        # - state: observations
        # - action: chosen action
        # - reward: immediate reward
        # - next_state: state after action
        # - done: completed state

        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        # save the current step to the memory
        # if hit the UPDATE_EVERY and we have enough memory, we train a bit
        # input:
        # - state: observations
        # - eps: how greedy we perform 0 greedy 1 random
        # output:
        # - action: chosen action

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        # we learn from experiences updatint the network
        # input:
        # - experiences: previous steps sampled from memory
        # - gamma: discount of future rewards

        states, actions, rewards, next_states, dones = experiences

        Q_target_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_target_next * (1-dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        # we slowly update the target to avoid moving target
        # input:
        # - local_model:
        # - target_model:
        # - tau: update factor 0 no update 1 complete update

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)
