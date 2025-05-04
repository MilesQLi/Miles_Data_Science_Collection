import torch
import torch.optim as optim
import gymnasium as gym
import random
import wandb
import time
import argparse
from collections import deque
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# nn.SiLU works but nn.SiLU does not work

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def get_sample(self, batch_size):
        samples = random.sample(list(self.buffer), batch_size)
        states, actions, next_states, rewards, dones = zip(
            *samples
        )
        return states, actions, next_states, rewards, dones


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.act_fn1 = nn.SiLU()
        self.layer_2 = nn.Linear(256, 128)
        self.act_fn2 = nn.SiLU()
        self.layer_3 = nn.Linear(128, action_dim)
        self.max_action = max_action


    def forward(self, x):
        x = self.act_fn1 (self.layer_1(x))
        x = self.act_fn2(self.layer_2(x))
        x = self.layer_3(x)
        #x = x + torch.randn_like(x) * (self.noise_scale * (self._step/self.max_noise_steps))
        x = self.max_action * torch.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.act_fn1 = nn.SiLU()
        self.layer_2 = nn.Linear(256, 128)
        self.act_fn2 = nn.SiLU()
        self.layer_3 = nn.Linear(128, 1)

    def forward(self, x, u):
        x = self.act_fn1(self.layer_1(torch.cat([x, u], 1)))
        x = self.act_fn2(self.layer_2(x))
        x = self.layer_3(x)
        return x

class Actor_SAC(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor_SAC, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.act_fn1 = nn.SiLU()
        self.layer_2 = nn.Linear(256, 128)
        self.act_fn2 = nn.SiLU()
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = self.act_fn1(self.layer_1(x))
        x = self.act_fn2(self.layer_2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor_SAC(state_dim, action_dim, max_action).cuda()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=1e-4)

        self.critic1 = Critic(state_dim, action_dim).cuda()
        self.critic_optimizer1 = optim.AdamW(self.critic1.parameters(), lr=1e-4)

        self.critic_target1 = Critic(state_dim, action_dim).cuda()
        self.critic_target1.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(state_dim, action_dim).cuda()
        self.critic_optimizer2 = optim.AdamW(self.critic2.parameters(), lr=1e-4)

        self.critic_target2 = Critic(state_dim, action_dim).cuda()
        self.critic_target2.load_state_dict(self.critic2.state_dict())


        self.max_action = max_action
        self._step = 0
        self.target_entropy = -action_dim

        self.target_entropy = -action_dim  # Default target entropy


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=3e-4)
    def make_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        mean, log_std = self.actor(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)* self.max_action
        return action[0].detach().cpu().numpy()

    def sample_action_and_log_prob(self, state):
        mean, log_std = self.actor(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)* self.max_action
        log_pi = normal.log_prob(z) - torch.log(
            1 - action.pow(2) + 1e-6
        )
        log_pi = log_pi.sum(1, keepdim=True)
        return action, log_pi

    def train(self, replay_buffer, batch_size, gamma=0.99, tau=0.005):
        self.alpha = self.log_alpha.exp()
        states, actions, next_states, rewards, dones = replay_buffer.get_sample(batch_size)
        states = torch.FloatTensor(np.array(states)).cuda()
        actions = torch.FloatTensor(np.array(actions)).cuda()
        next_states = torch.FloatTensor(np.array(next_states)).cuda()
        rewards = torch.FloatTensor(np.array(rewards)).cuda().unsqueeze(1)
        dones = torch.FloatTensor(np.array(dones)).cuda().unsqueeze(1)

        # Update Critic
        with torch.no_grad():
            next_actions, next_log_pi = self.sample_action_and_log_prob(next_states)
            Q1_target = self.critic_target1(next_states, next_actions)
            Q2_target = self.critic_target2(next_states, next_actions)
            Q_target = torch.min(Q1_target, Q2_target) - self.alpha * next_log_pi
            Q_target = rewards + gamma * (1 - dones) * Q_target

        Q1 = self.critic1(states, actions)
        Q2 = self.critic2(states, actions)
        critic_loss1 = nn.MSELoss()(Q1, Q_target)
        critic_loss2 = nn.MSELoss()(Q2, Q_target)
        #wandb.log({"critic_loss1": critic_loss1, "critic_loss2": critic_loss2})
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        # Update Actor
        actions, log_pi = self.sample_action_and_log_prob(states)
        Q1 = self.critic1(states, actions)
        Q2 = self.critic2(states, actions)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha * log_pi - Q).mean()
        #wandb.log({"actor_loss": actor_loss})
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # Update Target Networks
        for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)


class DDPG:
    def __init__(self, state_dim, action_dim, max_action, max_noise_steps=1000000):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).cuda()
        self.critic_target = Critic(state_dim, action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self._step = max_noise_steps
        self.max_noise_steps = max_noise_steps
        self.noise_scale = 0.2
        self.max_action = max_action

        self.ou_noise = OUActionNoise(
            mean=np.zeros(action_dim), std_deviation=float(0.2) * np.ones(action_dim)
        )
    def make_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.actor(state).cpu().data.numpy().flatten()
        #noise = np.random.normal(0, self.noise_scale * (self._step / self.max_noise_steps) * self.max_action, size=action.shape)
        action = action + self.ou_noise()#+ noise
        #print("Action: ", action)
        #print("Noise: ", noise)

        action = np.clip(action, -self.max_action, self.max_action)

        self._step -= 1
        self._step = max(0, self._step)
        return action

    def train(self,replay_buffer, batch_size, gamma=0.99, tau=0.005):
        states, actions, next_states, rewards, dones = replay_buffer.get_sample(batch_size)
        states = torch.FloatTensor(states).cuda()
        actions = torch.FloatTensor(actions).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        rewards = torch.FloatTensor(rewards).cuda()
        dones = torch.FloatTensor(dones).cuda()

        Q_local = self.critic(states, actions)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_next_target = self.critic_target(next_states, next_actions)

            Q_target = rewards + gamma * Q_next_target * (1 - dones)

        critic_loss = nn.MSELoss()(Q_local, Q_target)
        #wandb.log({"critic_loss": critic_loss})
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        #wandb.log({"actor_loss": actor_loss})
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)





def save_video(agent, episode, env_name):
    import imageio
    env = gym.make(env_name, render_mode='rgb_array')
    frames = []
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.make_action(state)
        next_state, _, done, _, _ = env.step(action)
        frames.append(env.render())
        state = next_state
    env.close()
    filename = f"bipedal_episode_{episode}.gif"
    imageio.mimsave(filename, frames, fps=30)
    print(f"Video saved as {filename}")



#=============
# Basic Setup
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="BipedalWalker-v3")
parser.add_argument(
    "--algorithm", default="DDPG", choices=["DDPG", "SAC"]
)
args = parser.parse_args()


env_name = args.env
algorithm = args.algorithm
wandb.init(project=env_name, name="my_"+algorithm)
seed = 0
max_timesteps = 50000000
start_timesteps = 10000
batch_size = 256
eval_freq = 5000
replay_buffer = ReplayBuffer(1000000)
max_steps_per_episode = 1000
save_video_every = 100

env = gym.make(env_name)#, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

print("State Dim: ", state_dim)
print("Action Dim: ", action_dim)
print("Max Action: ", max_action)

if algorithm == "DDPG":
    policy = DDPG(state_dim, action_dim, max_action)
elif algorithm == "SAC":
    policy = SAC(state_dim, action_dim, max_action)
#=============



last_10_rewards = deque(maxlen=10)
state, _ = env.reset()
current_steps = 0
episode_index = 1
episode_reward = 0
for i in tqdm(range(max_timesteps)):
    if i < start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.make_action(state)
    current_steps += 1
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    episode_reward += reward

    replay_buffer.add(state, action, next_state, reward, done)

    state = next_state
    if done or current_steps >= max_steps_per_episode:
        last_10_rewards.append(episode_reward)
        state, _ = env.reset()
        current_steps = 0
        avg_last_10_rewards = np.mean(last_10_rewards)
        wandb.log({"reward": episode_reward,'AVG_Reward': avg_last_10_rewards},step = episode_index)
        print("Episode:",episode_index, "Episode Reward: ", episode_reward)
        episode_index += 1
        episode_reward = 0

        if episode_index % save_video_every == 0:
            save_video(policy, episode=episode_index, env_name=env_name)

    if i > start_timesteps:
        policy.train(replay_buffer, batch_size)

torch.save(policy.actor, f"{env}_{algorithm}_policy_actor.pth")
torch.save(policy.critic, f"{env}_{algorithm}_policy_critic.pth")


