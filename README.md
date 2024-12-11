# RL-final-project
Mastering Continuous Control: Comparing A2C, SAC, and TD3 in OpenAI Gym

This project evaluates the performance of three reinforcement learning (RL) methods:

- **Advantage Actor-Critic (A2C)**
- **Soft Actor-Critic (SAC)**
- **Twin Delayed Deep Deterministic Policy Gradient (TD3)**

on continuous-action OpenAI Gym environments, **Pendulum** and **BipedalWalker**.

## Project Overview

The goal of this project is to compare on-policy and off-policy RL methods in terms of:

- **Sample Efficiency**: How efficiently each method learns from collected data. i.e. Learning speed
- **Stability of Learning**: The consistency of performance during training.
- **Robustness**: Performance in challenging continuous control tasks.

### Methods Evaluated

1. **A2C**: An on-policy method with synchronous policy updates.
2. **SAC**: An off-policy method leveraging entropy-based exploration.
3. **TD3**: An off-policy method with targeted action-value optimization.

### Environments

We test these methods on two OpenAI Gym environments:

- **Pendulum**: A simpler task with lower-dimensional state and action spaces.
- **BipedalWalker**: A dynamic, high-dimensional task requiring more complex control.


## Repository Contents

- **`bin/`**: Implementation of A2C, SAC, and TD3 algorithms.
- **`environment.yml`**: Environment file for conda env creation
- **`results/`**: Training logs, performance plots, and evaluation metrics.
- **`notebooks/`**: Jupyter notebooks for analysis and visualization.

## Getting Started

```bash
conda env create -f environment.yml
```
```bash
python train.py --env Pendulum-v1 --method SAC

```
