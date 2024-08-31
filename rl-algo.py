import gymnasium as gym
import os
from stable_baselines3.common.vec_env import SubprocVecEnv

# Register the environment
gym.envs.registration.register(
    id='ServerFleetEnv',
    entry_point='custom-rl-env:ServerFleetEnv',
    max_episode_steps=168,
)

env = gym.make('ServerFleetEnv')
num_cpu = os.cpu_count()

