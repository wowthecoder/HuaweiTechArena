import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from utils import load_problem_data, save_solution, save_json
from evaluation import get_actual_demand
from seeds import known_seeds
from custom_rl_env import format_actions

# For more examples, refer to https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

# Register the environment
gym.envs.registration.register(
    id='ServerFleetEnv',
    entry_point='custom_rl_env:ServerFleetEnv',
    max_episode_steps=168,
)

# load the problem data
demands, datacenters, servers, selling_prices = load_problem_data()

demands = get_actual_demand(demands, seed=1061)
demands.to_csv('./rl_data/actual_demand_1061.csv', index=False)
num_cpu = os.cpu_count()

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, datacenters=datacenters, demands=demands, servers=servers, selling_prices=selling_prices)
        env.reset(seed=seed + rank)
        wrapped_env = FlattenObservation(env)
        check_env(wrapped_env)
        return env
    
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    # Create the model
    vec_env = SubprocVecEnv([make_env("ServerFleetEnv", i) for i in range(num_cpu)])
    model = PPO("MlpPolicy", vec_env, verbose=1)
    # Print the device
    print(f"Model is using device: {model.device}")

    # Create a checkpoint callback to save the model every 1680 steps (10 iterations)
    checkpoint_callback = CheckpointCallback(save_freq=1680, save_path='./rl-logs/', name_prefix='ppo_checkpoint')

    # To resume training from a checkpoint, uncomment the code below:
    # Directory where checkpoints are saved
    # checkpoint_dir = './rl-logs/'

    # # List all files in the checkpoint directory
    # checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]

    # # Sort checkpoint files by timestep (assuming the naming convention)
    # checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))

    # # Get the most recent checkpoint file
    # latest_checkpoint = checkpoint_files[-1]
    # latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    # # Load the most recent checkpoint
    # model = PPO.load(latest_checkpoint_path, env=env)

    # # Resume training
    # model.learn(total_timesteps=10000)

    # Train 100 times
    model.learn(total_timesteps=16800, callback=checkpoint_callback)
    model.save("ppo_model_v1")

    # Later, load the model and resume training
    # The model continues learning from where it left off
    # model = PPO.load("ppo_model_v1", env=env)
    # model.learn(total_timesteps=10000)

    obs, info = vec_env.reset()
    # Make a solution for each dictionary
    # Get the best score 
    training_seeds = known_seeds('training')
    print("\nNow predicting\n")
    for seed in training_seeds:
        objective = 0
        solution = []
        for _ in range(168):
            actions, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = vec_env.step(actions)
            actions = format_actions(actions)
            solution.append(actions)
            objective += reward
            vec_env.render()
            # print a divider
            print("--" * 20)

        save_solution(solution, f"./output/{seed}.json")
        
        print("Objective:", objective)

