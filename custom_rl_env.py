import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box, Tuple, MultiDiscrete # Be careful, Tuple represents a tuple of variable length
import pandas as pd
from evaluation import get_time_step_demand, get_time_step_fleet, update_fleet, get_capacity_by_server_generation_latency_sensitivity, \
    check_datacenter_slots_size_constraint, get_utilization, get_normalized_lifespan, get_profit, put_fleet_on_hold, \
    change_selling_prices_format, solution_data_preparation

num_server_gens = 7
num_timesteps = 168
max_servers = 28000
max_demands_per_timestep = num_server_gens

def map_action(action):
    action_map = ["buy", "move", "dismiss"]
    sgen_map = ["CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4", "GPU.S1", "GPU.S2", "GPU.S3"]
    if action[4] != 3: # If the action is not "hold"
        return {
            "time_step": action[0],
            "datacenter_id": f"DC{action[1]}",
            "server_generation": sgen_map[action[2]],
            "server_id": action[3],
            "action": action_map[action[4]]
        }
    
    return None

class ServerFleetEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def _init_state(self, datacenters, demands, servers, selling_prices):
        self.datacenters = datacenters
        self.time_step = 1

        self.servers = servers
        self.selling_prices = selling_prices
        self.formatted_selling_prices = change_selling_prices_format(selling_prices)

        self.demands = demands

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
        self.action_space = MultiDiscrete([num_timesteps + 1, 4, num_server_gens, 28000, 4, 2])

        # Pad the entries with zero because gym.spaces.Sequence is not supported by stable baselines3
        # Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)
        # Hence we omit the cost column and map the cost_of_energy column to integers in ascending order
        self.observation_space = Dict({
            # demand data
            "demand_generation": MultiDiscrete([num_server_gens] * max_demands_per_timestep),
            "high": MultiDiscrete([int(1e6)] * max_demands_per_timestep),
            "medium": MultiDiscrete([int(1e6)] * max_demands_per_timestep),
            "low": MultiDiscrete([int(1e6)] * max_demands_per_timestep),
            "time_step": Discrete(num_timesteps),
            # fleet state
            "datacenter_id": MultiDiscrete([4] * max_servers),
            "server_generation": MultiDiscrete([num_server_gens] * max_servers),
            "server_id": MultiDiscrete([max_servers] * max_servers),
            "action": MultiDiscrete([4] * max_servers),
            "server_type": MultiDiscrete([2] * max_servers),
            "release_time_1": MultiDiscrete([num_timesteps + 1] * max_servers),
            "release_time_2": MultiDiscrete([num_timesteps + 1] * max_servers),
            "purchase_price": MultiDiscrete([20000] * max_servers),
            "slots_size": MultiDiscrete([5] * max_servers),
            "energy_consumption": MultiDiscrete([5000] * max_servers),
            "capacity": MultiDiscrete([160] * max_servers),
            "life_expectancy": Discrete(97), # all 96
            "cost_of_moving": Discrete(1001), # all 1000
            "average_maintenance_fee": MultiDiscrete([3100] * max_servers),
            "cost_of_energy": MultiDiscrete([4] * max_servers), # only 4 possible float values
            "latency_sensitivity": MultiDiscrete([3] * max_servers),
            "slots_capacity": MultiDiscrete([26000] * max_servers),
            "selling_price": MultiDiscrete([3000] * max_servers),
            "lifespan": MultiDiscrete([100] * max_servers),
            "moved": MultiDiscrete([2] * max_servers),
        })

    def _get_obs(self):
        obs = {}
        sgen_map = {"CPU.S1": 0, "CPU.S2": 1, "CPU.S3": 2, "CPU.S4": 3, "GPU.S1": 4, "GPU.S2": 5, "GPU.S3": 6}

        demand_ts = self.demands.loc[self.demands["time_step"] == self.time_step].copy() 
        demand_ts["demand_generation"] = demand_ts["server_generation"].map(sgen_map)
        # format: {'server_generation': ['CPU.S1', 'GPU.S1'], 'high': [172, 83], 'low': [36228, 13], 'medium': [5735, 35]}
        demand_dict = demand_ts.drop(columns=["time_step", "server_generation"], inplace=False).to_dict(orient='list')
        # Pad the demand_list with empty dicts
        padding = [0] * (max_demands_per_timestep - len(demand_ts))
        for k in demand_dict.keys():
            obs[k] = demand_dict[k] + padding

        obs["time_step"] = self.time_step 
        obs["life_expectancy"] = 96
        obs["cost_of_moving"] = 1000

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
            }
        else:
            # Convert all the values in Fleet to numerical
            dcid_map = {"DC1": 1, "DC2": 2, "DC3": 3, "DC4": 4}
            action_map = {"buy": 0, "move": 1, "dismiss": 2, "hold": 4}
            stype_map = {"CPU": 0, "GPU": 1}
            latency_map = {"low": 0, "medium": 1, "high": 2}
            energycost_map = {0.25: 0, 0.35: 1, 0.65: 1, 0.75: 1}
            obs2 = self.fleet.copy()
            obs2["datacenter_id"] = obs2["datacenter_id"].map(dcid_map)
            obs2["server_generation"] = obs2["server_generation"].map(sgen_map)
            obs2["action"] = obs2["action"].map(action_map)
            obs2["server_type"] = obs2["server_type"].map(stype_map)
            obs2["latency_sensitivity"] = obs2["latency_sensitivity"].map(latency_map)
            obs2[["release_time1", "release_time2"]] = obs2["release_time"].str.strip('[]').str.split(',', expand=True)
            obs2["cost_of_energy"] = obs2["cost_of_energy"].map(energycost_map)
            obs2.drop(columns=["release_time", "life_expectancy", "cost"], inplace=True)
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
        sgen_map = ["CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4", "GPU.S1", "GPU.S2", "GPU.S3"]
        dcid, sgen, sid, act = action[1], action[2], action[3], action[4]
        rtimes = list(map(int, self.servers.loc[sgen, 'release_time'].strip('[]').split(',')))
        # center = self.data_centers[dcid]
        # server_info = self.server_info[sgen]
        if (act == 0 and sid in self.fleet["server_ids"]) \
        or (act == 1 and sid not in self.fleet["server_ids"]) \
        or (act == 2 and sid not in self.fleet["server_ids"]) \
        or action[0] != self.time_step \
        or (act == 0 and self.time_step not in rtimes) \
        or (act != 0 and sgen_map[sgen] != self.fleet.loc[self.fleet["server_id"] == sid, 'server_generation'].values[0]):
            return False
        
        return True 

    def calculate_reward(self):
        # GET THE ACTUAL DEMAND AT TIMESTEP ts
        D = get_time_step_demand(self.demands, self.time_step)
  
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
        truncated = False  # we do not limit the number of steps here

        # Check if action is valid
        # if not self.is_action_valid(action):
        #     # Optionally, penalize the agent for selecting an invalid action
        #     reward = -10.0
        #     if action[5] == 1:
        #         self.time_step += 1
        #     return self._get_obs[], reward, terminated, truncated, {}

        # GET THE SERVERS DEPLOYED AT TIMESTEP ts
        # Check if constraints are obeyed
        try: 
            mapped_action = map_action(action)
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
            return self._get_obs(), reward, terminated, truncated, {}

        reward = self.calculate_reward()

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        if action[5] == 1:
            self.time_step += 1

        return (self._get_obs(), reward, terminated, truncated, info)

    def render(self):
        print("Timestep:", self.time_step)
        print("State:", self._get_obs())
        print("Fleet:", self.fleet)

    def close(self):
        pass