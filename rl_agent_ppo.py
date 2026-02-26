"""
PPO (Proximal Policy Optimization) Agent for Adaptive Emergency Vehicle Detection.

Philosophy: PPO is more stable and sample-efficient than DQN, with better convergence
on continuous and discrete action spaces. Perfect for parameter tuning + alert decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque
import os


class PPONetwork(nn.Module):
    """Actor-Critic network: Shared backbone with separate policy and value heads."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, state: torch.Tensor) -> tuple:
        """Returns logits (for policy) and value estimate."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value


class PPOAgent:
    """PPO Agent with trajectory collection and multi-epoch updates."""
    
    def __init__(
        self, 
        state_size: int, 
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 3,
        batch_size: int = 64,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 PPO Agent using device: {self.device}")
        
        # Network
        self.network = PPONetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Metrics
        self.step_count = 0
        self.episode_reward = 0.0
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray) -> tuple:
        """
        Select action using learned policy (with exploration via sampling).
        Returns: (action, value_estimate, log_prob)
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.network.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device.startswith('cuda')):
                policy_logits, value = self.network(state_t)
        
        # Sample from policy distribution
        dist = Categorical(logits=policy_logits)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        value_est = value.item()
        
        self.network.train()
        return action, value_est, log_prob
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store trajectory for batch update."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        self.episode_reward += reward
        self.step_count += 1
    
    def reset_episode(self):
        """Reset episode counter."""
        self.episode_reward = 0.0
        self.step_count = 0
    
    def compute_gae(self, next_value: float) -> tuple:
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        returns = []
        gae = 0.0
        
        # Process trajectory in reverse
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return np.array(advantages), np.array(returns)
    
    def update(self, next_state: np.ndarray):
        """Perform multi-epoch PPO update on collected trajectory."""
        if len(self.states) == 0:
            return
        
        # Compute returns and advantages
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.network(next_state_t)
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Convert to tensors
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        
        # Multi-epoch update
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                
                # Forward pass
                policy_logits, values = self.network(batch_states)
                
                # Policy loss (PPO objective with clipping)
                dist = Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus (encourages exploration)
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear trajectory buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        self.total_steps += n_samples
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.network.state_dict(), path)
        print(f"✅ PPO model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.to(self.device)
        print(f"✅ PPO model loaded from {path}")
