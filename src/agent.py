"""
DQN and DDQN agents implemented from scratch in PyTorch.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─── Network ─────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ─── Base Agent ───────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 200_000,
        batch_size: int = 64,
        target_update: int = 100,
        buffer_size: int = 10_000,
        device: str = "cpu",
        double: bool = False,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device)
        self.double = double
        self.steps = 0

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray) -> int:
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.steps / self.epsilon_decay
        )
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(s).argmax(dim=1).item())

    def push(self, *args):
        self.buffer.push(*args)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.tensor(s,  device=self.device)
        a  = torch.tensor(a,  device=self.device)
        r  = torch.tensor(r,  device=self.device)
        s2 = torch.tensor(s2, device=self.device)
        d  = torch.tensor(d,  device=self.device)

        q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                # Double DQN: select action with policy net, evaluate with target net
                best_actions = self.policy_net(s2).argmax(dim=1, keepdim=True)
                q_next = self.target_net(s2).gather(1, best_actions).squeeze(1)
            else:
                q_next = self.target_net(s2).max(dim=1).values
            q_target = r + self.gamma * q_next * (1 - d)

        loss = nn.functional.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


class DDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, double=True, **kwargs)
