import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random

# Hyperparameters
state_dim = 3  # Example state dimension
action_dim = 1  # Example action dimension
max_action = 1.0
discount = 0.99
tau = 0.005
alpha = 0.2
batch_size = 256
learning_rate = 3e-4
replay_buffer_capacity = 1_000_000
max_iterations = 1_000_000

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t) * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        q_value = self.l3(x)
        return q_value

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)

# Initialize environment
env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize networks and optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(state_dim, action_dim, max_action).to(device)
critic_1 = Critic(state_dim, action_dim).to(device)
critic_2 = Critic(state_dim, action_dim).to(device)
target_critic_1 = Critic(state_dim, action_dim).to(device)
target_critic_2 = Critic(state_dim, action_dim).to(device)
target_critic_1.load_state_dict(critic_1.state_dict())
target_critic_2.load_state_dict(critic_2.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_1_optimizer = optim.Adam(critic_1.parameters(), lr=learning_rate)
critic_2_optimizer = optim.Adam(critic_2.parameters(), lr=learning_rate)

# Initialize replay buffer
replay_buffer = ReplayBuffer(replay_buffer_capacity)

# Training loop
state, done = env.reset(), False
episode_reward = 0
for iteration in range(max_iterations):
    # Select action
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action, log_prob = actor.sample(state_tensor)
    action = action.cpu().detach().numpy()[0]

    # Perform action
    next_state, reward, done, _ = env.step(action)
    replay_buffer.add(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward

    if done:
        state, done = env.reset(), False
        print(f"Iteration: {iteration}, Episode Reward: {episode_reward}")
        episode_reward = 0

    # Update networks if enough samples are available
    if replay_buffer.size() > batch_size:
        # Sample a batch from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        # Move to device
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)
        done_batch = done_batch.to(device)

        # Compute target Q-value
        with torch.no_grad():
            next_action, next_log_prob = actor.sample(next_state_batch)
            target_q1 = target_critic_1(next_state_batch, next_action)
            target_q2 = target_critic_2(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * discount * target_q

        # Update Critic networks
        current_q1 = critic_1(state_batch, action_batch)
        current_q2 = critic_2(state_batch, action_batch)
        critic_1_loss = nn.MSELoss()(current_q1, target_q)
        critic_2_loss = nn.MSELoss()(current_q2, target_q)

        critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_1_optimizer.step()

        critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        critic_2_optimizer.step()

        # Update Actor network
        new_action, log_prob = actor.sample(state_batch)
        q1_new = critic_1(state_batch, new_action)
        q2_new = critic_2(state_batch, new_action)
        actor_loss = (alpha * log_prob - torch.min(q1_new, q2_new)).
::contentReference[oaicite:0]{index=0}
 
