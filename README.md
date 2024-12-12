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
- **`RL-final-report-pavia.pdf`**: final report
- **`bin/`**: Implementation of A2C, SAC, and TD3 algorithms.
    - **`bin/logs/`**: Sample efficiency data
    - **`bin/output/`**: Training and Evaluation output
    - **`bin/tensorboard/`**: Tensorboard logs
- **`hyperparameters/`**: Hyperparameter YML from RL Baselines3 Zoo
- **`environment.yml`**: Environment file for conda env creation
- **`notebooks/`**: notebooks for analysis and visualization.
    - **`notebooks/continuous-control.ipynb`**: Main notebook for example training and evaluation

## Getting Started

```bash
conda env create -f environment.yml
```

```
conda activate py37
```

Running the repo via command line is the preferred method over the notebook
```bash
python train.py --env "Pendulum-v1" --method "SAC" --seed 0
```
To connect to tensorboard plotting during training
```
tensorboard --logdir ../notebooks/tensorboard/a2c_Pendulum-v1_seed1/
```

To run multiple experiments use run.sh
```
nohup ./run.sh > ./output/output.log 2>&1 &
```