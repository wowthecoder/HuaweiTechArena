import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Sequence, Box, Tuple # Be careful, sequence represents a tuple of variable length
import uuid

num_server_gens = 7

class DataCenterEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    def _init_state(self, datacenters, demands, servers, selling_prices):
        self.data_centers = []  # list of data centers
        self.time_step = 0
        for i in range(len(datacenters)):
            dc = {
                "total_slots": datacenters.loc[i, 'slots_capacity'],
                "remaining_slots": datacenters.loc[i, 'slots_capacity'],
                "energy_cost": datacenters.loc[i, 'cost_of_energy'],
                "latency_sensitivity": datacenters.loc[i, 'latency_sensitivity'],
                "servers": [],
                "server_ids": set()
            }
            self.data_centers.append(dc)

        # server info 
        # convert the release time from string to list of integers
        servers["release_time"] = servers["release_time"].apply(lambda x: list(map(int, x.strip('[]').split(','))))
        self.server_info = servers 
        self.selling_prices = selling_prices
        self.demands = demands
        

    # datacenters, servers, selling_prices are pandas dataframes
    def __init__(self, datacenters, demands, servers, selling_prices, render_mode="console"):
        super(DataCenterEnv, self).__init__()
        self.render_mode = render_mode

        self._init_state(datacenters, demands, servers, selling_prices)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = Sequence(Dict({
            "time_step": Discrete(168), 
            "datacenter_id": Discrete(4), # 0: DC1, 1: DC2, 2: DC3, 3: DC4
            "server_generation": Discrete(num_server_gens), # 0: CPU.S1, 1: CPU.S2, 2: CPU.S3, 3: CPU.S4, 4: GPU.S1, 5: GPU.S2, 6: GPU.S3
            "server_id": Discrete(28000), # max 28000 servers ( Total 55845 slots across all data centers, min 2 slots per server)
            "action": Discrete(3) # 0: buy, 1: move, 2: dismiss
        }))

        # The observation will be the state of the data centers
        server_data = Dict({
            "generation": Discrete(num_server_gens),
            "server_id": Discrete(28000),
            "operating_time": Discrete(96),
        })
        server_info = Dict({
            "generation": Discrete(num_server_gens),
            "release_time": Tuple(Discrete(168), Discrete(168)),
            "selling_price": Box(low=0, high=3000, shape=(1,), dtype=int),
            "purchase_price": Box(low=1, high=200000, shape=(1,), dtype=int),
            "slots_size": Box(low=1, high=5, shape=(1,), dtype=int),
            "energy_consumption": Box(low=1, high=5000, shape=(1,), dtype=int),
            "capacity": Box(low=1, high=200, shape=(1,), dtype=int),
            "life_expectancy": Discrete(1), # all 96
            "cost_of_moving": Discrete(1), # all 1000
        })
        center_data = Dict({
            "total_slots": Discrete(1),
            "remaining_slots": Box(low=0, high=26000, shape=(1,), dtype=int),
            "energy_cost": Box(low=0.0, high=1.0, shape=(1,), dtype=float),
            "latency_sensitivity": Discrete(3), # 0: low, 1: medium, 2: high
            "servers": Sequence(server_data)
        })
        demand_data = Dict({
            "time_step": Discrete(168),
            "latency_sensitivity": Discrete(3),
            "demands": Tuple([Box(low=0, high=1e6, shape=(1,), dtype=int)] * num_server_gens), # CPU.S1-4, GPU.S1-3
        })
        self.observation_space = Dict({
            "demand": Tuple([demand_data] * 3), # each step there's 3 demands, 1 for each latency sensitivity
            "time_step": Discrete(168),
            "server_info": Tuple([server_info] * num_server_gens),
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
    
    def is_action_valid(self, actions):
        # 1. Check if server id is valid ( buy needs to create a new id, move and dismiss need to use an existing id )
        # 2. Check if data center has enough slots left
        # 3. Check if server is available for purchase at that time step
        # 4. Check if server move is valid ( the source datacenter has the server and the destination datacenter has enough slots)

        # Combine all server ids
        combined_server_ids = {server_id for dc in self.data_centers for server_id in dc["server_ids"]}
        for action in actions:
            if (action["action"] == 0 and action["server_id"] in combined_server_ids) \
            or (action["action"] == 1 and action["server_id"] not in combined_server_ids) \
            or (action["action"] == 2 and action["server_id"] not in self.data_centers[action["datacenter_id"]]["server_ids"]) \
            or action["time_step"] != self.time_step \
            or self.data_centers[action["datacenter_id"]]["remaining_slots"] < self.server_info.loc[action["server_generation"], "slots_size"] \
            or action["action"] == "buy" and self.time_step not in self.server_info.loc[action["server_generation"], "release_time"] \
            or action["action"] == "move" and action["server_id"] in self.data_centers[action["datacenter_id"]]["server_ids"] \ 
            or action["action"] == "move" and self.data_centers[action["datacenter_id"]]["remaining_slots"] < self.server_info.loc[action["server_generation"], "slots_size"]:
                return False
        return True 

    def calculate_reward(self, demand, time_step):
        reward = 0
        for i in range(len(self.data_centers)):
            center = self.data_centers[i]
            for server in center["servers"]:
                if server["generation"] == demand["latency_sensitivity"]:
                    reward += demand["demands"][server["generation"]] * self.selling_prices.loc[time_step, server["generation"]]
        return reward

    def step(self, actions):
        self.time_step += 1
        # Check if action is valid
        if not self.is_action_valid(actions):
            # Optionally, penalize the agent for selecting an invalid action
            reward = -10.0
            terminated = bool(self.time_step == 168)
            truncated = False
            return self._get_obs, reward, terminated, truncated, {}

        terminated = bool(self.time_step == 168)
        truncated = False  # we do not limit the number of steps here

        # Apply all actions
        datacenter = self.data_centers[action["datacenter_id"]]
        for action in actions:
            if action["action"] == "buy":
                server = {
                    "generation": action["server_generation"],
                    "release_time": self.server_info.loc[action["server_generation"], "release_time"],
                    "server_id": action["server_id"],
                    "selling_price": self.selling_prices.loc[action["server_generation"] + 7 * datacenter["latency_sensitivity"], "selling_price"],
                    "purchase_price": self.server_info.loc[action["server_generation"], "purchase_price"],
                    "slots_size": self.server_info.loc[action["server_generation"], "slots_size"],
                    "energy_consumption": self.server_info.loc[action["server_generation"], "energy_consumption"],
                    "capacity": self.server_info.loc[action["server_generation"], "capacity"],
                    "life_expectancy": self.server_info.loc[action["server_generation"], "life_expectancy"],
                    "cost_of_moving": self.server_info.loc[action["server_generation"], "cost_of_moving"],
                    "operating_time": 0
                }
                # Add server id to the set
                self.server_ids.add(action["server_id"])
            elif action["action"] == "move":
                asdf
            else: # dismiss
                # Remove server id from set
                self.server_ids.remove(action["server_id"])

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