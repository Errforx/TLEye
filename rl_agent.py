import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingDQN(nn.Module):
    """Dueling DQN architecture for faster convergence."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        # Shared feature extraction
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Value stream
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Advantage stream
        self.advantage_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_head(x)
        advantage = self.advantage_head(x)

        # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # Avoid inplace operations: compute mean, subtract separately
        mean_advantage = advantage.mean(dim=1, keepdim=True)
        q = value + advantage - mean_advantage
        return q


class RLAgent:
    """Ultra-fast Dueling DQN agent with aggressive learning."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 5e-3,  # Aggressive learning rate
        gamma: float = 0.95,  # Focus on near-term rewards
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,  # Exploit sooner
        epsilon_decay: float = 0.985,  # Faster decay
        buffer_size: int = 50000,  # Huge replay buffer
        batch_size: int = 128,  # Large batches
        target_update_freq: int = 100,  # Update target often
        device: str | None = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Use Dueling DQN for faster convergence
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.update_target()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=lr, weight_decay=1e-5
        )
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0

    def update_target(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy (optimized for speed)."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Fast inference mode
        self.policy_net.eval()  # Disable training-specific operations
        state_t = (torch.tensor(state, dtype=torch.float32, device=self.device)
                   .unsqueeze(0))
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device.startswith('cuda')):
                qvals = self.policy_net(state_t)
        
        action = int(qvals.argmax().item())
        self.policy_net.train()  # Re-enable training mode
        return action

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train(self) -> None:
        """Train using Double DQN with Huber loss (optimized for real-time performance)."""
        if len(self.memory) < self.batch_size:
            return

        self.policy_net.train()  # Ensure train mode
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors on appropriate device
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute current Q-values
        curr_q = self.policy_net(states).gather(1, actions)

        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            expected_q = rewards + (1 - dones) * self.gamma * next_q

        # Huber loss (robust to outliers)
        loss = nn.SmoothL1Loss()(curr_q, expected_q)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target()

        # Decay epsilon for exploration-exploitation tradeoff
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target()
