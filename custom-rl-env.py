import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Text, Sequence, Box
import uuid

class DataCenterEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    def _init_state(self, datacenters, servers, selling_prices):
        self.server_ids = set() # list of server ids in the data center
        self.data_centers = []  # list of data centers
        for i in range(len(datacenters)):
            dc = {
                "total_slots": datacenters.loc[i, 'slots_capacity'],
                "occupied_slots": 0,
                "energy_cost": datacenters.loc[i, 'cost_of_energy'],
                "latency_sensitivity": datacenters.loc[i, 'latency_sensitivity'],
                "servers": []
            }
            self.data_centers.append(dc)

        # server info
        self.server_info = servers 
        self.selling_prices = selling_prices
        

    # datacenters, servers, selling_prices are pandas dataframes
    def __init__(self, datacenters, servers, selling_prices, render_mode="console"):
        super(DataCenterEnv, self).__init__()
        self.render_mode = render_mode

        self._init_state(datacenters, servers, selling_prices)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = Sequence(Dict({
            "time_step": Discrete(168), 
            "datacenter_id": Discrete(4), # 0: DC1, 1: DC2, 2: DC3, 3: DC4
            "server_generation": Discrete(7), # 0: CPU.S1, 1: CPU.S2, 2: CPU.S3, 3: CPU.S4, 4: GPU.S1, 5: GPU.S2, 6: GPU.S3
            "server_id": Discrete(28000), # max 28000 servers ( Total 55845 slots across all data centers, min 2 slots per server)
            "action": Discrete(3) # 0: buy, 1: move, 2: dismiss
        }))

        # The observation will be the state of the data centers
        server_data = Dict({
            "generation": Discrete(7),
            "release_time": Discrete(168),
            "server_id": Text(min_length=36, max_length=36),
            "selling_price": Box(low=0, high=3000, shape=(1,), dtype=int),
            "purchase_price": Box(low=1, high=200000, shape=(1,), dtype=int),
            "slots_size": Box(low=1, high=5, shape=(1,), dtype=int),
            "energy_consumption": Box(low=1, high=5, shape=(1,), dtype=int),
            "capacity": Box(low=1, high=200, shape=(1,), dtype=int),
            "life_expectancy": Discrete(1), # all 96
            "cost_of_moving": Discrete(1), # all 1000
            "operating_time": Discrete(96),
        })
        center_data = Dict({
            "total_slots": Discrete(1),
            "occupied_slots": Discrete(1),
            "energy_cost": Box(low=0.0, high=1.0, shape=(1,), dtype=float),
            "latency_sensitivity": Discrete(3), # 0: low, 1: medium, 2: high
            "servers": Sequence(server_data)
        })
        demand_data = Dict({
            "time_step": Discrete(168),
            "latency_sensitivity": Discrete(3),
            "demands": Sequence(Box(low=0, high=1e6, shape=(1,), dtype=int)), # CPU S1-4 and GPU S1-3
        })
        self.observation_space = Dict({
            "demand": demand_data,
            "time_step": Discrete(168),
            "DC1": center_data,
            "DC2": center_data,
            "DC3": center_data,
            "DC4": center_data
        })

    def _get_obs(self, demand={}, time_step=1):
        obs = {}
        obs["demand"] = demand
        obs["time_step"] = time_step
        for i in len(self.data_centers):
            obs["DC"+str(i+1)] = self.data_centers[i]
        return obs
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.server_ids = set() 
        for center in self.data_centers:
            center["occupied_slots"] = 0
            center["servers"] = []
        return self._get_obs({}, 1), {}  # empty info dict
    
    def is_action_valid(self, action):
        return action["action"] == "buy" or action["server_id"] in self.server_ids

    def calculate_reward(self, demand, time_step):
        reward = 0
        for i in range(len(self.data_centers)):
            center = self.data_centers[i]
            for server in center["servers"]:
                if server["generation"] == demand["latency_sensitivity"]:
                    reward += demand["demands"][server["generation"]] * self.selling_prices.loc[time_step, server["generation"]]
        return reward

    def step(self, action):
        # Check if action is valid
        if not self.is_action_valid(action):
            # Optionally, penalize the agent for selecting an invalid action
            reward = -10.0
            terminated = bool(action["time_step"] == 167)
            truncated = False
            return self._get_obs, reward, terminated, truncated, {}

        terminated = bool(action["time_step"] == 167)
        truncated = False  # we do not limit the number of steps here

        # Apply action
        datacenter = self.data_centers[action["datacenter_id"]]
        # Automatically dismiss any server that is past the life expectancy

        reward = self.calculate_reward()

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        print("State: ", self.data_centers)

    def close(self):
        pass