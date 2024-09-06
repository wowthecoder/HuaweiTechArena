import pandas as pd
import gymnasium as gym
import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import PPO
from utils import save_solution, load_problem_data
from seeds import known_seeds
from custom_rl_env import map_action
from evaluation import get_actual_demand

orig_demands, datacenters, servers, selling_prices = load_problem_data()
# demands = get_actual_demand(demands, seed=1061)

gym.envs.registration.register(
    id='ServerFleetEnv',
    entry_point='custom_rl_env:ServerFleetEnv',
    max_episode_steps=30000,
)

# Make a solution for each dictionary
# Get the best score 
# To resume training from a checkpoint, uncomment the code below:
# Directory where checkpoints are saved
checkpoint_dir = './rl_logs/mask_ppo_v2'

# List all files in the checkpoint directory
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]

# Sort checkpoint files by timestep (assuming the naming convention)
checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))

# Get the most recent checkpoint file
latest_checkpoint = checkpoint_files[-1]
latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
print(latest_checkpoint_path)

# Make a solution for each dictionary
# Get the best score 
training_seeds = known_seeds('training')
print("\nNow predicting\n")
for seed in training_seeds:
    demands = get_actual_demand(orig_demands, seed=seed)
    env = gym.make("ServerFleetEnv", datacenters=datacenters, demands=demands, servers=servers, selling_prices=selling_prices)
    # Load the most recent checkpoint
    model = MaskablePPO.load(latest_checkpoint_path, env=env)
    obs, info = env.reset()
    objective = 0
    solution = []
    timestep = 1
    while timestep < 169:
        action, _states = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        action = map_action(action, timestep)
        nextstep = action.pop("nextstep")
        if action["action"] != "hold" and info["valid"]:
            solution.append(action)
        timestep += nextstep 
        objective += reward
        print(action, info["valid"])
        # print a divider
        print("--" * 20)
        if terminated or truncated:
            print("terminated at timestep", timestep, terminated, truncated)
            break

    save_solution(solution, f"./test_output/{seed}.json")
    
    print(f"Objective for seed {seed} is: {objective}")