# BipedalWalker-v3 Implementation with DDPG and SAC

This repository contains an implementation of two state-of-the-art reinforcement learning algorithms, Deep Deterministic Policy Gradient (DDPG) and Soft Actor-Critic (SAC), applied to the BipedalWalker-v3 environment from OpenAI Gymnasium.

## Overview

The BipedalWalker-v3 environment is a challenging continuous control task where an agent must learn to walk using a bipedal robot. The implementation includes:

- DDPG (Deep Deterministic Policy Gradient) algorithm
- SAC (Soft Actor-Critic) algorithm
- Experience replay buffer
- Ornstein-Uhlenbeck noise for exploration
- Neural network architectures for both actor and critic networks
- Training visualization and video recording

## Features

- **Dual Algorithm Support**: Choose between DDPG and SAC implementations
- **Experience Replay**: Efficient memory buffer for storing and sampling experiences
- **Noise Injection**: Ornstein-Uhlenbeck noise for better exploration
- **Neural Network Architecture**:
  - Actor network with SiLU activation functions
  - Critic network with SiLU activation functions
  - Target networks for stable learning
- **Training Monitoring**: Integration with Weights & Biases (wandb) for tracking training progress
- **Video Recording**: Automatic saving of agent performance videos during training

## Requirements

- Python 3.x
- PyTorch
- Gymnasium
- NumPy
- Weights & Biases (wandb)
- imageio (for video recording)

## Usage

To train the agent, run the following command:

```bash
python ActorCritic.py --env BipedalWalker-v3 --algorithm [DDPG|SAC]
```

### Arguments

- `--env`: Environment name (default: "BipedalWalker-v3")
- `--algorithm`: Algorithm to use (choices: "DDPG" or "SAC")

## Training Progress

The training progress can be monitored through:
- Console output showing episode rewards
- Weights & Biases dashboard
- Saved video recordings of the agent's performance

Here's a demonstration of the trained agent in action:

![BipedalWalker Demo](bipedal_episode_1000.gif)

## Implementation Details

### DDPG Algorithm
- Uses deterministic policy gradient
- Implements target networks for stability
- Employs Ornstein-Uhlenbeck noise for exploration
- Uses Adam optimizer for both actor and critic networks

### SAC Algorithm
- Implements entropy-regularized reinforcement learning
- Uses automatic temperature tuning
- Maintains two Q-functions to reduce overestimation bias
- Employs target networks for stability

## Results

The implementation achieves stable walking behavior in the BipedalWalker-v3 environment. The agent learns to:
- Maintain balance
- Coordinate leg movements
- Navigate the terrain
- Optimize energy usage

## Future Improvements

Potential areas for improvement:
- Hyperparameter optimization
- Implementation of additional algorithms (e.g., TD3)
- Enhanced exploration strategies
- Curriculum learning for more efficient training
