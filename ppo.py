from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class RolloutBuffer:
    obs: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logprobs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)

    def extend(self, other: "RolloutBuffer") -> None:
        self.obs.extend(other.obs)
        self.actions.extend(other.actions)
        self.logprobs.extend(other.logprobs)
        self.values.extend(other.values)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()


def compute_gae(
    rewards,
    dones,
    values,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    """Compute generalized advantage estimation for one rollout."""
    advantages = []
    gae = 0.0
    next_value = 0.0

    for step in reversed(range(len(rewards))):
        mask = 0.0 if dones[step] else 1.0
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[step]

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns


def ppo_update(
    model,
    optimizer,
    buffer: RolloutBuffer,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    ppo_epochs: int = 4,
    minibatch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 0.5,
):
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    obs = torch.stack(buffer.obs)
    device = obs.device
    actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_logprobs = torch.stack(buffer.logprobs).detach().to(device)
    old_values = torch.stack(buffer.values).detach().view(-1).cpu().tolist()

    advantages, returns = compute_gae(
        rewards=buffer.rewards,
        dones=buffer.dones,
        values=old_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages = advantages.to(device)
    returns = returns.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    n = obs.shape[0]
    minibatch_size = min(minibatch_size, n)
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(ppo_epochs):
        indices = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start:start + minibatch_size]

            new_logprobs, entropy, new_values = model.evaluate_actions(obs[batch_idx], actions[batch_idx])
            new_values = new_values.view(-1)

            ratio = torch.exp(new_logprobs - old_logprobs[batch_idx])
            surr1 = ratio * advantages[batch_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages[batch_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_values, returns[batch_idx])
            entropy_bonus = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_bonus.item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
    }
