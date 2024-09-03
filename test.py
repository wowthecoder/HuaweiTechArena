import pandas as pd
import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import PPO
from utils import save_solution, load_problem_data
from seeds import known_seeds
from custom_rl_env import map_action
from evaluation import get_actual_demand

demands, datacenters, servers, selling_prices = load_problem_data()
demands = get_actual_demand(demands, seed=1061)

gym.envs.registration.register(
    id='ServerFleetEnv',
    entry_point='custom_rl_env:ServerFleetEnv',
    max_episode_steps=168,
)

env = gym.make("ServerFleetEnv", datacenters=datacenters, demands=demands, servers=servers, selling_prices=selling_prices)
obs, info = env.reset()
# Make a solution for each dictionary
# Get the best score 
# To resume training from a checkpoint, uncomment the code below:
# Directory where checkpoints are saved
checkpoint_dir = './rl-logs/'

# List all files in the checkpoint directory
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]

# Sort checkpoint files by timestep (assuming the naming convention)
checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))

# Get the most recent checkpoint file
latest_checkpoint = checkpoint_files[-1]
latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
print(latest_checkpoint_path)

# Load the most recent checkpoint
model = PPO.load(latest_checkpoint_path, env=env)
training_seeds = known_seeds('training')
print("\nNow predicting\n")
for seed in training_seeds:
    objective = 0
    solution = []
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        action = map_action(action)
        if action: # if action is not hold
            solution.append(action)
            print(action)
        objective += reward
        # print a divider
        print("--" * 20)
        if terminated or truncated:
            break

    save_solution(solution, f"./test_output/{seed}.json")
    
    print(f"Objective for seed {seed} is: {objective}")