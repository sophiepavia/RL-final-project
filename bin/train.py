import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold



class SampleEfficiencyCallback(BaseCallback):
    """
    Callback to track rewards and total environment steps during training.
    Logs sample efficiency for later analysis.
    """
    def __init__(self, log_dir, verbose=0):
        super(SampleEfficiencyCallback, self).__init__(verbose)
        self.rewards = []
        self.timesteps = []
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        # Log the episode reward and total timesteps
        if "episode" in self.locals["infos"][0]:
            self.rewards.append(self.locals["infos"][0]["episode"]["r"])
            self.timesteps.append(self.num_timesteps)
        return True

    def save_results(self):
        # Save results for plotting
        np.savetxt(os.path.join(self.log_dir, "sample_efficiency.csv"),
                   np.column_stack((self.timesteps, self.rewards)),
                   header="timesteps,rewards", delimiter=",", comments="")
        
def load_hyperparameters(algo_name, env_name):
    """
    Load hyperparameters from RL Baselines3 Zoo configuration files.
    """
    algo_name = algo_name.lower()
    config_path = f"../hyperparameters/{algo_name}.yml"
    with open(config_path, "r") as file:
        hyperparams = yaml.safe_load(file)

    # Get the specific hyperparameters for the environment
    if env_name in hyperparams:
        hyperparams = hyperparams[env_name]
        return hyperparams

def plot_sample_efficiency(log_dir):
    """
    Plot sample efficiency: Rewards vs. Environment Steps.
    """
    data = np.loadtxt(os.path.join(log_dir, "sample_efficiency.csv"), delimiter=',', skiprows=1)
    timesteps, rewards = data[:, 0], data[:, 1]

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards, label="Sample Efficiency")
    plt.xlabel("Environment Steps")
    plt.ylabel("Episodic Rewards")
    plt.title("Sample Efficiency: Rewards vs Environment Steps")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(log_dir, "sample_efficiency.png"))
    # plt.show()

def evaluate(model, num_episodes=10, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    vec_env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = vec_env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            # also note that the step only returns a 4-tuple, as the env that is returned
            # by model.get_env() is an sb3 vecenv that wraps the >v0.26 API
            obs, reward, done, info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward  

def train(method, env_name, hyperparams, tensor_board_dir, log_dir):
    # For make_vec_env
    n_envs = hyperparams.pop("n_envs", 1)
    normalize = hyperparams.pop("normalize", False)
    
    vec_env = make_vec_env(env_name, n_envs=n_envs)
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create a callback to log sample efficiency
    sample_callback = SampleEfficiencyCallback(log_dir=log_dir)
    
    if env_name == "BipedalWalker-v3":
        # Stop training when the model reaches the reward threshold
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
        eval_callback = EvalCallback(vec_env, callback_on_new_best=callback_on_best, verbose=1)
    
        callback = CallbackList([sample_callback, eval_callback])
    elif env_name == "Pendulum-v1":
        callback = sample_callback
        
    
    # Not in all hyperparameter files
    policy_kwargs = hyperparams.pop("policy_kwargs", None)
    if isinstance(policy_kwargs, str):
        policy_kwargs = eval(policy_kwargs)
    
    learning_rate = hyperparams.pop("learning_rate", None)
    
    # Convert learning rate to a callable function
    if isinstance(learning_rate, str):
        if learning_rate.startswith("lin_"):
            # Extract the numerical part after "lin_"
            lr_value = float(learning_rate.split("_")[1])
            learning_rate = get_schedule_fn(lr_value)
        else:
            raise ValueError(f"Unsupported learning rate format: {learning_rate}")
    
    policy = hyperparams.pop("policy", None)
    
    # For model.learn
    n_timesteps = hyperparams.pop("n_timesteps", None)
    
    # Select the model
    if method == "a2c":
        model = A2C(policy=policy, env=vec_env, verbose=1, tensorboard_log=tensor_board_dir, policy_kwargs=policy_kwargs, learning_rate=learning_rate, **hyperparams)
    elif method == "sac":
        model = SAC(policy=policy, env=vec_env, verbose=1, tensorboard_log=tensor_board_dir, policy_kwargs=policy_kwargs, learning_rate=learning_rate, **hyperparams)
    elif method == "td3":
        noise_type = hyperparams.pop("noise_type", "normal")
        
        if noise_type == "normal":
            n_actions = vec_env.action_space.shape[-1]
            noise_std = hyperparams.pop("noise_std", None)
            hyperparams["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
            
        elif noise_type == "ornstein_uhlenbeck":
            print("Warning: Ornstein-Uhlenbeck noise is not implemented.")
            
        model = TD3(policy=policy, env=vec_env, verbose=1, tensorboard_log=tensor_board_dir, policy_kwargs=policy_kwargs, learning_rate=learning_rate, **hyperparams)
        
    else:
        raise ValueError(f"Unsupported method: {method}, please choose from 'A2C', 'SAC', or 'TD3'.")

    # Train the model
    print(f"Training {method} on {env_name}...")
    if n_timesteps is not None:
        model.learn(n_timesteps, callback=callback)
    else:
        print("Error: n_timesteps not specified.")
    
    # Save the results
    sample_callback.save_results()
    
    return model, vec_env, normalize
        
def save_model(model, method, env_name, normalize, vec_env, log_dir):
    # Save the model
    model.save(f"../models/{method.lower()}_{env_name}")
    if normalize:
        vec_env.save(f"../models/{method.lower()}_{env_name}_vec_normalize.pkl")
    print("Training complete. Model saved.")
    
    # Plot sample efficiency
    print("Plotting sample efficiency...")
    plot_sample_efficiency(log_dir)

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train an RL model on a specified environment.")
    parser.add_argument("--env", type=str, required=True, help="Name of the OpenAI Gym environment. (Please use Pendulum or BipedalWalker)")
    parser.add_argument("--method", type=str, required=True, help="RL method to use (A2C, SAC, or TD3).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set the seed
    seed = args.seed
    print(f"=== Training with seed: {seed} ===")
    set_seed(seed)

    # Logging directories
    env_name = args.env
    method = args.method.lower()
    tensor_board_dir = f"./tensorboard/{method}_{env_name}_seed{seed}/"
    os.makedirs(tensor_board_dir, exist_ok=True)
    log_dir = f"./logs/{method}_{env_name}_seed{seed}"
    os.makedirs(log_dir, exist_ok=True)

    # Load hyperparameters
    hyperparams = load_hyperparameters(method, env_name)

    # Train and save the model
    model, vec_env, normalize = train(method, env_name, hyperparams, tensor_board_dir, log_dir)
    save_model(model, method, env_name, normalize, vec_env, log_dir)

    # Evaluate the model
    print("Evaluating model...")
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=100)
    print(f"Seed {seed}: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    

if __name__ == "__main__":
    main()
