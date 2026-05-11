import torch
import torch.nn as nn
from torch.distributions import Categorical


class LidarGoalActorCritic(nn.Module):
    def __init__(self, in_channels: int = 6, scan_bins: int = 96, num_actions: int = 5):
        super().__init__()
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )

        feat_dim = 64 * 16
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        x = self.encoder(obs)
        x = self.backbone(x)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def policy_stats(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        probs = dist.probs
        entropy = dist.entropy()
        if probs.shape[-1] >= 2:
            top2 = torch.topk(probs, k=2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
        else:
            margin = torch.ones_like(entropy)
        return dist, value, entropy, margin, probs

    def act(self, obs: torch.Tensor, return_stats: bool = False):
        dist, value, entropy, margin, probs = self.policy_stats(obs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        if return_stats:
            return action, logprob, value, entropy, margin, probs
        return action, logprob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, entropy, value
