import torch

import numpy as np
import random
from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        # ReplayBuffer stores the memory of the previous steps and
        # it is used to randomly sampling the previous experiences
        # input:
        # - action_size: number of possible actions
        # - buffer_size: total size of memory buffer
        # - batch_size: number of replays returned during sampling
        # - seed: for reproducability
        # - device: device to be used

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        # input:
        # - state: observations
        # - action: chosen action
        # - reward: immediate reward
        # - next_state: state after action
        # - done: completed state

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # output:
        # - list of (states, actions, rewards, next_states, dones), of size batch_size, randomly sampled from the memory

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
