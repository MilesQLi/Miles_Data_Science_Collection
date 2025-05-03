import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import gymnasium as gym
import wandb

# --- Configuration ---
ENV_NAME = "LunarLander-v3"
WANDB_PROJECT = "dqn-lunarlander" 
WANDB_ENTITY = None 


class DQN(nn.Module):
    def __init__(self, state_size, action_size, device='cpu'):
        super(DQN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.ffn(x).squeeze(0)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, sample):
        self.buffer.append(sample)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*samples)
        return (torch.FloatTensor(state), torch.LongTensor(action),
                torch.FloatTensor(reward), torch.FloatTensor(next_state),
                torch.FloatTensor(done))

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_size, action_size, buffer_capacity=10000, batch_size=64, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, learning_rate=0.001, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.learning_rate = learning_rate
        self.device = device

        self.steps_done = 0

        self.policy_net = DQN(state_size, action_size, device)
        self.target_net = DQN(state_size, action_size, device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def select_action(self, state):
        sample = random.random()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if sample > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return torch.argmax(self.policy_net(state)).item()
        else:
            return random.randrange(self.action_size)
        
    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        next_state = torch.tensor(next_state).unsqueeze(0).to(self.device)
        done = torch.tensor(done).to(self.device)

        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_state_values = self.target_net(next_state).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma * (1 - done)) + reward

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()
    
    def soft_update(self, tau=0.005):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def update_target_network(self):
        self.soft_update()


def save_video(agent, episode):
    import imageio
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    frames = []
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, _, done, _, _ = env.step(action)
        frames.append(env.render())
        state = next_state
    env.close()
    filename = f"lunarlander_dqn_episode_{episode}.gif"
    imageio.mimsave(filename, frames, fps=30)
    print(f"Video saved as {filename}")

# Setup the environment and agent

number_of_episodes = 3000
maximum_steps_per_episode = 1000
save_video_every = 100

wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,name='dqn-lunarlander-myDQN')
env = gym.make(ENV_NAME)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(state_size, action_size, device=device)
rolling_reward = deque(maxlen=50)
global_step = 0

# Training loop

for i in range(1, number_of_episodes + 1):
    state,_  = env.reset()
    total_reward = 0
    for t in range(maximum_steps_per_episode):
        global_step += 1
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.buffer.push((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        agent.optimize_model()

        if done:
            break
    

    rolling_reward.append(total_reward)
    wandb.log({"episode": i, 'epsilon': agent.epsilon, "cumulative_reward": total_reward, "rolling_average_reward": np.mean(rolling_reward), "steps_this_episode": t}, step=global_step)
    agent.steps_done += 1
    if i % 10 == 0:
        print(f"Episode {i}, Total Reward: {total_reward}, Rolling Average Reward: {np.mean(rolling_reward)}")
    if i % save_video_every == 0:
        save_video(agent, episode=i)
        wandb.log({"video": wandb.Video(f"lunarlander_dqn_episode_{i}.gif")}, step=global_step)


save_video(agent, filename="lunarlander_dqn.gif")