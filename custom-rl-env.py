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
            "server_id": Text(min_length=36, max_length=36), # uuid4 is a 36 character string, including 4 hyphens
            "action": Discrete(3) # 0: buy, 1: move, 2: dismiss
        }))

        # The observation will be the state of the data centers
        server_data = Dict({
            "generation": Discrete(7),
            "release_time": Discrete(168),
            "server_id": Text(min_length=36, max_length=36),
            "selling_price": Discrete(1),
            "purchase_price": Discrete(1),
            "slots_size": Discrete(1),
            "energy_consumption": Discrete(1),
            "capacity": Discrete(1),
            "life_expectancy": Discrete(1),
            "cost_of_moving": Discrete(1),
            "operating_time": Discrete(1),
        })
        center_data = Dict({
            "total_slots": Discrete(1),
            "occupied_slots": Discrete(1),
            "energy_cost": Box(low=0.0, high=1.0, shape=(1,), dtype=float),
            "latency_sensitivity": Discrete(3), # 0: low, 1: medium, 2: high
            "servers": Sequence(server_data)
        })
        demand_data = Dict({

        })
        self.observation_space = Dict({
            "demand": demand_data,
            "time_step": Discrete(168),
            "DC1": center_data,
            "DC2": center_data,
            "DC3": center_data,
            "DC4": center_data
        })

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        # Initialize the agent at the right of the grid
        self.agent_pos = self.grid_size - 1
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32), {}  # empty info dict
    
    def get_valid_actions(self):
        pass

    def step(self, action):
        # Check if action is valid
        valid_actions = self.get_valid_actions()

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        terminated = bool(self.agent_pos == 0)
        truncated = False  # we do not limit the number of steps here

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array([self.agent_pos]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print("." * self.agent_pos, end="")
            print("x", end="")
            print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass