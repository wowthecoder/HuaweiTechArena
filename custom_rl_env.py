import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box, Tuple, MultiDiscrete # Be careful, Tuple represents a tuple of variable length
import pandas as pd
from evaluation import get_time_step_demand, get_time_step_fleet, update_fleet, get_capacity_by_server_generation_latency_sensitivity, \
    check_datacenter_slots_size_constraint, get_utilization, get_normalized_lifespan, get_profit, put_fleet_on_hold, \
    change_selling_prices_format, solution_data_preparation

num_server_gens = 7
num_timesteps = 168
max_servers = 20000
max_demands_per_timestep = num_server_gens
max_num_actions = 30000

def map_action(action):
    action_map = ["buy", "move", "dismiss"]
    sgen_map = ["CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4", "GPU.S1", "GPU.S2", "GPU.S3"]
    # Scale it back to the original discrete action space
    # Convert from [-1, 1] to [0, 1]
    scaled_action = (action + 1) / 2
    action = [
        round(scaled_action[0] * (num_timesteps - 1)),  # Map to [0, num_timesteps-1]
        round(scaled_action[1] * 3),  # Map to [0, 3] (4 possible values)
        round(scaled_action[2] * (num_server_gens - 1)),  # Map to [0, num_server_gens - 1]
        round(scaled_action[3] * (max_servers - 1)),  # Map to [0, 27999] (28000 possible values)
        round(scaled_action[4] * 2),  # Map to [0, 2] (3 possible values)
    ]
    return {
        "time_step": action[0] + 1,
        "datacenter_id": f"DC{action[1]}",
        "server_generation": sgen_map[action[2]],
        "server_id": action[3],
        "action": action_map[action[4]]
    }

class ServerFleetEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def _init_state(self, datacenters, demands, servers, selling_prices):
        self.datacenters = datacenters.copy()
        self.time_step = 1
        self.num_actions = 0 # used for truncation

        self.servers = servers.copy()
        self.selling_prices = selling_prices.copy()
        self.formatted_selling_prices = change_selling_prices_format(selling_prices)

        self.demands = demands.copy()
        sgen_map = {"CPU.S1": 0, "CPU.S2": 1, "CPU.S3": 2, "CPU.S4": 3, "GPU.S1": 4, "GPU.S2": 5, "GPU.S3": 6}
        d2 = demands.copy()
        d2["demand_generation"] = d2["server_generation"].map(sgen_map)
        d2.drop(columns="server_generation", inplace=True)
        self.normalised_demands = (d2 - d2.min()) / (d2.max() - d2.min())

        self.fleet = pd.DataFrame()
        

    # datacenters, demands, servers, selling_prices are pandas dataframes
    def __init__(self, datacenters, demands, servers, selling_prices, render_mode="console"):
        super(ServerFleetEnv, self).__init__()
        self.render_mode = render_mode

        self._init_state(datacenters, demands, servers, selling_prices)

        # Define action and observation space
        # They must be gym.spaces objects
        # Hold action is included because a Tuple must return at least one action, but sometimes we want to do nothing at a time step
        # Dictionary action space is not supported, but leaving the commented code here for description 
        # self.action_space = Dict({
        #     "time_step": Discrete(168), 
        #     "datacenter_id": Discrete(4), # 0: DC1, 1: DC2, 2: DC3, 3: DC4
        #     "server_generation": Discrete(num_server_gens), # 0: CPU.S1, 1: CPU.S2, 2: CPU.S3, 3: CPU.S4, 4: GPU.S1, 5: GPU.S2, 6: GPU.S3
        #     "server_id": Discrete(28000), # max 28000 servers ( Total 55845 slots across all data centers, min 2 slots per server)
        #     "action": Discrete(4) # 0: buy, 1: move, 2: dismiss, 3: hold
        #     "continue": Discrete(2) # 0: stay in current time step, 1: next time step
        # }))
        # According to the Tips and Tricks page of Stable Baselines3 docs, it is recommended to rescale action space to [-1, 1]
        # self.action_space = MultiDiscrete([num_timesteps + 1, 4, num_server_gens, max_servers, 4, 2])
        self.action_space = Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32), 
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        )

        # Pad the entries with zero because gym.spaces.Sequence is not supported by stable baselines3
        # Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)
        # Normalise every column, remove constant columns like life_expectancy and cost_of_moving
        demand_col = Box(low=0, high=1, shape=(max_demands_per_timestep,), dtype=np.float32)
        fleet_col = Box(low=0, high=1, shape=(max_servers,), dtype=np.float32)
        self.observation_space = Dict({
            # demand data
            "demand_generation": demand_col,
            "high": demand_col,
            "medium": demand_col,
            "low": demand_col,
            "time_step": Discrete(num_timesteps),
            # fleet state
            "datacenter_id": fleet_col,
            "server_generation": fleet_col,
            "server_id": fleet_col,
            "action": fleet_col,
            "server_type": fleet_col,
            "release_time_1": fleet_col,
            "release_time_2": fleet_col,
            "purchase_price": fleet_col,
            "slots_size": fleet_col,
            "energy_consumption": fleet_col,
            "capacity": fleet_col,
            "average_maintenance_fee": fleet_col,
            "cost_of_energy": fleet_col,
            "latency_sensitivity": fleet_col,
            "slots_capacity": fleet_col,
            "selling_price": fleet_col,
            "lifespan": fleet_col,
            "moved": fleet_col,
            "cost": fleet_col,
        })

    def _get_obs(self):
        obs = {}

        demand_ts = self.demands.loc[self.demands["time_step"] == self.time_step].copy() 
        normalised_demand_ts = self.normalised_demands.loc[demand_ts.index].copy()
        # format: {'server_generation': ['CPU.S1', 'GPU.S1'], 'high': [172, 83], 'low': [36228, 13], 'medium': [5735, 35]}
        demand_dict = normalised_demand_ts.drop(columns=["time_step"], inplace=False).to_dict(orient='list')
        # Pad the demand_list with empty dicts
        padding = [0] * (max_demands_per_timestep - len(demand_ts))
        for k in demand_dict.keys():
            obs[k] = demand_dict[k] + padding

        obs["time_step"] = self.time_step 

        # If fleet is empty (just after reset, return an empty dictionary)
        if self.fleet.empty:
            obs2 = {
                "datacenter_id": [],
                "server_generation": [],
                "server_id": [],
                "action": [],
                "server_type": [],
                "release_time_1": [],
                "release_time_2": [],
                "purchase_price": [],
                "slots_size": [], # either 2 or 4
                "energy_consumption": [],
                "capacity": [],
                "average_maintenance_fee": [],
                "cost_of_energy": [],
                "latency_sensitivity": [],
                "slots_capacity": [],
                "selling_price": [],
                "lifespan": [],
                "moved": [],
                "cost": []
            }
        else:
            # Convert all the values in Fleet to numerical
            dcid_map = {"DC1": 1, "DC2": 2, "DC3": 3, "DC4": 4}
            sgen_map = {"CPU.S1": 0, "CPU.S2": 1, "CPU.S3": 2, "CPU.S4": 3, "GPU.S1": 4, "GPU.S2": 5, "GPU.S3": 6}
            action_map = {"buy": 0, "move": 1, "dismiss": 2, "hold": 4}
            stype_map = {"CPU": 0, "GPU": 1}
            latency_map = {"low": 0, "medium": 1, "high": 2}
            obs2 = self.fleet.copy()
            obs2["datacenter_id"] = obs2["datacenter_id"].map(dcid_map)
            obs2["server_generation"] = obs2["server_generation"].map(sgen_map)
            obs2["action"] = obs2["action"].map(action_map)
            obs2["server_type"] = obs2["server_type"].map(stype_map)
            obs2["latency_sensitivity"] = obs2["latency_sensitivity"].map(latency_map)
            obs2[["release_time1", "release_time2"]] = obs2["release_time"].str.strip('[]').str.split(',', expand=True)
            obs2[["release_time1", "release_time2"]] = obs2[["release_time1", "release_time2"]].astype(int)
            obs2.drop(columns=["release_time", "life_expectancy", "cost_of_moving"], inplace=True)
            # Normalise all columns
            obs2 = (obs2 - obs2.min()) / (obs2.max() - obs2.min())
            obs2 = obs2.to_dict(orient='list')

        # Pad all entries with zeroes up to max_servers
        padding = [0] * (max_servers - len(self.fleet))
        for k in obs2.keys():
            obs[k] = obs2[k] + padding

        return obs
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.time_step = 1
        self.fleet = pd.DataFrame()
        # need to include the first demand in the first observation
        return self._get_obs(), {}
    
    def is_action_valid(self, action):
        # 1. Check if server id is valid ( buy needs to create a new id, move needs the id to not exist in destination datacenter, dismiss need to use an existing id )
        # 3. Check if server is available for purchase at that time step
        # 4. Check if server move is valid ( the source datacenter has the server and the destination datacenter has enough slots)
        # 5. If move/dismiss, check if server generation matches id

        # Combine all server ids
        # combined_server_ids = {server_id for dc in self.data_centers for server_id in dc["servers_dict"].keys[]}
        dcid, sgen, sid, act = action["datacenter_id"], action["server_generation"], action["server_id"], action["action"]
        rtimes = list(map(int, self.servers.loc[sgen, 'release_time'].strip('[]').split(',')))
        # center = self.data_centers[dcid]
        # server_info = self.server_info[sgen]
        if (act == 0 and sid in self.fleet["server_ids"]) \
        or (act == 1 and sid not in self.fleet["server_ids"]) \
        or (act == 2 and sid not in self.fleet["server_ids"]) \
        or action["time_step"] != self.time_step \
        or (act == 0 and self.time_step not in rtimes) \
        or (act != 0 and sgen != self.fleet.loc[self.fleet["server_id"] == sid, 'server_generation'].values[0]):
            return False
        
        return True 

    def calculate_reward(self):
        # GET THE ACTUAL DEMAND AT TIMESTEP ts
        D = get_time_step_demand(self.demands, self.time_step)
        reward = 0
  
        # CHECK IF THE FLEET IS EMPTY
        if self.fleet.shape[0] > 0:
            # GET THE SERVERS CAPACITY AT TIMESTEP ts
            Zf = get_capacity_by_server_generation_latency_sensitivity(self.fleet)

            # EVALUATE THE OBJECTIVE FUNCTION AT TIMESTEP ts
            U = get_utilization(D, Zf)
    
            L = get_normalized_lifespan(self.fleet)
    
            P = get_profit(D, 
                           Zf, 
                           self.formatted_selling_prices,
                           self.fleet)
            reward = U * L * P

            # PUT ENTIRE FLEET on HOLD ACTION
            self.fleet = put_fleet_on_hold(self.fleet)
        
        return reward

    def step(self, action):
        terminated = bool(self.time_step == num_timesteps)
        truncated = self.num_actions > max_num_actions
        self.num_actions += 1

        # Check if action is valid
        # if not self.is_action_valid(action):
        #     # Optionally, penalize the agent for selecting an invalid action
        #     reward = -10.0
        #     if action[5] == 1:
        #         self.time_step += 1
        #     return self._get_obs[], reward, terminated, truncated, {}

        # GET THE SERVERS DEPLOYED AT TIMESTEP ts
        # Check if constraints are obeyed
        mapped_action = map_action(action)
        if not self.is_action_valid(mapped_action):
            reward = -10.0
            if action[5] > 0:
                self.time_step += 1
            return self._get_obs(), reward, terminated, truncated, {}
        try:
            solution = solution_data_preparation(pd.DataFrame(mapped_action), self.servers, self.datacenters, self.selling_prices)
            ts_fleet = get_time_step_fleet(solution, self.time_step)

            if ts_fleet.empty and not self.fleet.empty:
                ts_fleet = self.fleet
            elif ts_fleet.empty and self.fleet.empty:
                return 0

            # UPDATE FLEET
            new_fleet = update_fleet(self.time_step, self.fleet, ts_fleet)

            check_datacenter_slots_size_constraint(new_fleet)
            self.fleet = new_fleet 
        except ValueError as ve:
            reward = -10.0
            if action[5] > 0:
                self.time_step += 1
            return self._get_obs(), reward, terminated, truncated, {}

        reward = self.calculate_reward()

        if action[5] > 0:
            self.time_step += 1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (self._get_obs(), reward, terminated, truncated, info)

    def render(self):
        print("Timestep:", self.time_step)
        print("State:", self._get_obs())
        print("Fleet:", self.fleet)

    def close(self):
        pass