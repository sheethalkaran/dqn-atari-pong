# DQN Atari Pong

Implementation of a **Deep Q-Network (DQN)** agent trained to play the Atari game **Pong** using **Stable-Baselines3**. The agent leverages convolutional neural networks (CNNs) to learn optimal gameplay strategies through reinforcement learning. The trained model achieves near-perfect performance in Pong.


## Overview

The DQN agent interacts with the `PongNoFrameskip-v4` environment from the Atari suite. It uses a combination of **frame stacking**, **epsilon-greedy exploration**, and **target network updates** to efficiently learn optimal actions.

The training pipeline records episode rewards and evaluates the agent’s performance using multiple episodes after training.


## Features

- Implementation of DQN with `CnnPolicy` for image-based input.  
- Frame stacking to provide temporal context to the agent.  
- Reward logging with moving average visualization.  
- Model evaluation on unseen episodes.  
- Model saving and loading for reuse.  


## Environment Setup

To run this project, you need:

- Python 3.8+  
- **Stable-Baselines3**  
- **Gymnasium** with Atari support  
- **AutoROM** to download Atari ROMs  
- **NumPy** and **Matplotlib** for reward tracking and visualization  
Install dependencies with:
```pip install stable-baselines3[extra] gymnasium[atari] autorom```

Run ```AutoROM --accept-license``` to install required Atari ROMs.

## Training

The agent is trained using the following key parameters:

- **Total timesteps:** 1,500,000  
- **Replay buffer size:** 50,000  
- **Batch size:** 32  
- **Learning rate:** 0.0001  
- **Gamma (discount factor):** 0.99  
- **Exploration fraction:** 0.3  
- **Final epsilon:** 0.01  
- **Target network update interval:** 1,000  

During training, episode rewards are logged and plotted to visualize the learning progression.


## Evaluation

After training, the agent is evaluated over multiple episodes. Metrics reported include:

- **Mean reward**  
- **Standard deviation of reward**  

In this implementation, the agent achieves a **mean reward of 20**, indicating near-optimal Pong performance.


## Results

- **Number of episodes:** 1080  
- **Min reward:** -21  
- **Max reward:** 21  
- **Mean reward:** -4.09  
- **Evaluation mean reward:** 20.00 ± 0.00  

A reward progression plot shows improvement over episodes with a moving average trend line.