"""
A2C and PPO agents for the single-asset TradingEnv.

These agents are intentionally written to match the existing DQN training loop:
- select_action(state, greedy=False)
- push(state, action, reward, next_state, done)
- update()
- current_epsilon()

Unlike DQN/DDQN, they are on-policy actor-critic methods. They collect one full
training episode, update when the terminal transition is received, and then clear
the rollout buffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """Shared MLP trunk with a categorical actor head and scalar critic head."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value


@dataclass
class RolloutBuffer:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def push(self, state, action, reward, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)


class A2CAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        normalize_advantage: bool = True,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.device = torch.device(device)
        self.steps = 0

        self.policy_net = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def current_epsilon(self) -> float:
        """Compatibility with the existing DQN training printout."""
        return 0.0

    def greedy_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            logits, _ = self.policy_net(s)
            return int(logits.argmax(dim=-1).item())

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if greedy:
            return self.greedy_action(state)

        s = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        logits, _ = self.policy_net(s)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item())

    def push(self, state, action, reward, next_state, done):
        # next_state is not needed for Monte-Carlo-return A2C/PPO here, but the
        # signature matches DQNAgent.push so src.evaluate.run_episode can be reused.
        self.buffer.push(state, action, reward, done)

    def _returns_tensor(self) -> torch.Tensor:
        returns = []
        running_return = 0.0

        for reward, done in zip(
            reversed(self.buffer.rewards),
            reversed(self.buffer.dones),
        ):
            if done:
                running_return = 0.0
            running_return = reward + self.gamma * running_return
            returns.append(running_return)

        returns.reverse()
        return torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

    def _batch_tensors(self):
        states = torch.tensor(
            np.asarray(self.buffer.states, dtype=np.float32),
            device=self.device,
        )
        actions = torch.tensor(
            np.asarray(self.buffer.actions, dtype=np.int64),
            device=self.device,
        )
        returns = self._returns_tensor()
        return states, actions, returns

    def update(self):
        # The existing training loop calls update() after every step. For on-policy
        # learning, wait until the episode is complete, then update once.
        if len(self.buffer) == 0 or not bool(self.buffer.dones[-1]):
            return None

        states, actions, returns = self._batch_tensors()

        logits, values = self.policy_net(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        advantages = returns - values

        if self.normalize_advantage and advantages.numel() > 1:
            detached_adv = advantages.detach()
            adv_for_actor = (
                detached_adv - detached_adv.mean()
            ) / (detached_adv.std() + 1e-8)
        else:
            adv_for_actor = advantages.detach()

        actor_loss = -(log_probs * adv_for_actor).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            self.max_grad_norm,
        )
        self.optimizer.step()

        self.steps += len(self.buffer)
        loss_value = float(loss.item())
        self.buffer.clear()
        return loss_value


class PPOAgent(A2CAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        normalize_advantage: bool = True,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantage=normalize_advantage,
            device=device,
        )
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    def update(self):
        if len(self.buffer) == 0 or not bool(self.buffer.dones[-1]):
            return None

        states, actions, returns = self._batch_tensors()

        with torch.no_grad():
            old_logits, old_values = self.policy_net(states)
            old_dist = Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions)

            advantages = returns - old_values
            if self.normalize_advantage and advantages.numel() > 1:
                advantages = (
                    advantages - advantages.mean()
                ) / (advantages.std() + 1e-8)

        n = states.shape[0]
        last_loss = None

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)

            for start in range(0, n, self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]

                logits, values = self.policy_net(states[idx])
                dist = Categorical(logits=logits)

                log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs[idx])
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                )

                policy_loss = -torch.min(
                    ratio * advantages[idx],
                    clipped_ratio * advantages[idx],
                ).mean()

                value_loss = nn.functional.mse_loss(values, returns[idx])

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                last_loss = float(loss.item())

        self.steps += len(self.buffer)
        self.buffer.clear()
        return last_loss