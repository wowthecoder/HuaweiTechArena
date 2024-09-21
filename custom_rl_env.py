import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box, Tuple, MultiDiscrete # Be careful, Tuple represents a tuple of variable length
import pandas as pd
from evaluation import get_time_step_demand, get_time_step_fleet, update_fleet, get_capacity_by_server_generation_latency_sensitivity, \
    check_datacenter_slots_size_constraint, get_utilization, get_normalized_lifespan, get_profit, put_fleet_on_hold, \
    change_selling_prices_format, solution_data_preparation
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

num_datacenters = 4
num_server_gens = 7
num_timesteps = 168
max_servers = 12000
max_demands_per_timestep = num_server_gens
# max_num_consec_buys = 30000
invalid_reward = int(-1e8)

class ServerFleetEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def _init_vars(self):
        self.time_step = 1
        self.num_consec_buys = 0
        self.ulp = [0, 0, 0]
        self.fleet = pd.DataFrame()

    def _init_state(self, datacenters, demands, servers, selling_prices):
        self._init_vars()

        self.datacenters = datacenters.copy()

        self.servers = servers.copy()
        self.selling_prices = selling_prices.copy()
        self.formatted_selling_prices = change_selling_prices_format(selling_prices)

        self.demands = demands.copy()
        sgen_map = {"CPU.S1": 1, "CPU.S2": 2, "CPU.S3": 3, "CPU.S4": 4, "GPU.S1": 5, "GPU.S2": 6, "GPU.S3": 7}
        d2 = demands.copy()
        d2["demand_generation"] = d2["server_generation"].map(sgen_map)
        d2.drop(columns="server_generation", inplace=True)
        # If we normalise as below, the min value of each column will be 0
        # we want to differentiate between padding and actual value, so just div by max()
        # self.normalised_demands = (d2 - d2.min()) / (d2.max() - d2.min())
        d2["demand_generation"] /= num_server_gens
        d2["high"] /= d2["high"].max()
        d2["medium"] /= d2["medium"].max()
        d2["low"] /= d2["low"].max()
        self.normalised_demands = d2

        rtime_copy = servers['release_time'].copy()

        # Apply the transformation to the copied column
        rtime_copy = rtime_copy.apply(lambda x: eval(x))

        # Flatten the lists and collect unique values
        self.rtimes = set([item for sublist in rtime_copy for item in sublist])

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
        #     "datacenter_id": Discrete(4), # 0: DC1, 1: DC2, 2: DC3, 3: DC4
        #     "server_generation": Discrete(num_server_gens), # 0: CPU.S1, 1: CPU.S2, 2: CPU.S3, 3: CPU.S4, 4: GPU.S1, 5: GPU.S2, 6: GPU.S3
        #     "server_id": Discrete(28000), # max 28000 servers ( Total 55845 slots across all data centers, min 2 slots per server)
        #     "action": Discrete(4) # 0: buy, 1: move, 2: dismiss, 3: hold
        #     "continue": Discrete(2) # 0: stay in current time step, 1: next time step
        # }))
        # According to the Tips and Tricks page of Stable Baselines3 docs, it is recommended to rescale action space to [-1, 1]
        # we update the timestep ourselves, don't let the model learn
        # self.action_space = Box(
        #     low=np.array([-1, -1, -1, -1, -1], dtype=np.float32), 
        #     high=np.array([1, 1, 1, 1, 1], dtype=np.float32)
        # )
        # MaskablePPO does not support Box action spaces, so we revert to MultiDiscrete
        dims = [num_datacenters, num_server_gens, max_servers, 4, 2]
        self.action_space = MultiDiscrete(dims)
        self.action_mask = [True] * sum(dims)

        # Pad the entries with zero because gym.spaces.Sequence is not supported by stable baselines3
        # Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)
        # Normalise every column, remove constant columns like life_expectancy and cost_of_moving
        demand_col = Box(low=0, high=1, shape=(max_demands_per_timestep,), dtype=np.float64)
        fleet_col = Box(low=0, high=1, shape=(max_servers,), dtype=np.float64)
        self.observation_space = Dict({
            # demand data
            "demand_generation": demand_col,
            "high": demand_col,
            "medium": demand_col,
            "low": demand_col,
            "time_step": Discrete(num_timesteps+1),
            # fleet state
            "datacenter_id": fleet_col,
            "server_generation": fleet_col,
            "server_id": fleet_col,
            # "release_time1": fleet_col,
            # "release_time2": fleet_col,
            "purchase_price": fleet_col,
            # "slots_size": fleet_col,
            "energy_consumption": fleet_col,
            # "capacity": fleet_col,
            "average_maintenance_fee": fleet_col,
            "cost_of_energy": fleet_col,
            "latency_sensitivity": fleet_col,
            "slots_capacity": fleet_col,
            "selling_price": fleet_col,
            "lifespan": fleet_col,
            "cost": Box(low=0, high=1e6, shape=(max_servers,), dtype=np.float64),
            "utilization": Box(low=0, high=1, shape=(1,), dtype=np.float64),
            "normalised_lifespan": Box(low=0, high=1, shape=(1,), dtype=np.float64),
            "profit": Box(low=0, high=1e8, shape=(1,), dtype=np.float64),
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
            obs[k] = np.array(demand_dict[k] + padding)

        obs["time_step"] = self.time_step

        # If fleet is empty (just after reset, return an empty dictionary)
        zeros = np.zeros(shape=(max_servers,))
        if self.fleet.empty:
            obs["datacenter_id"] = zeros
            obs["server_generation"] = zeros
            obs["server_id"] = zeros
            # obs["release_time1"] = zeros
            # obs["release_time2"] = zeros
            obs["purchase_price"] = zeros
            # obs["slots_size"] = zeros # either 2 or 4
            obs["energy_consumption"] = zeros
            # obs["capacity"] = zeros
            obs["average_maintenance_fee"] = zeros
            obs["cost_of_energy"] = zeros
            obs["latency_sensitivity"] = zeros
            obs["slots_capacity"] = zeros
            obs["selling_price"] = zeros
            obs["lifespan"] = zeros
            obs["cost"] = zeros
            
        else:
            # Convert all the values in Fleet to numerical
            dcid_map = {"DC1": 1, "DC2": 2, "DC3": 3, "DC4": 4}
            sgen_map = {"CPU.S1": 1, "CPU.S2": 2, "CPU.S3": 3, "CPU.S4": 4, "GPU.S1": 5, "GPU.S2": 6, "GPU.S3": 7}
            latency_map = {"low": 1, "medium": 2, "high": 3}
            obs2 = self.fleet.head(max_servers).copy()
            # Map to numeric values and normalise all columns except cost (becuz we dk the max and min values)
            # also cost_of_energy is already within 0 and 1
            obs2["datacenter_id"] = (obs2["datacenter_id"].map(dcid_map)) / num_datacenters
            obs2["server_generation"] = (obs2["server_generation"].map(sgen_map)) / num_server_gens
            obs2["server_id"] /= max_servers
            obs2["latency_sensitivity"] = (obs2["latency_sensitivity"].map(latency_map)) / 3
            # obs2[["release_time1", "release_time2"]] = obs2["release_time"].str.strip('[]').str.split(',', expand=True)
            # obs2[["release_time1", "release_time2"]] = (obs2[["release_time1", "release_time2"]].astype(int)) / 168
            obs2["purchase_price"] /= 160000
            # obs2["slots_size"] /= 4
            obs2["energy_consumption"] /= 4200
            # obs2["capacity"] /= 160
            obs2["average_maintenance_fee"] /= 3080
            obs2["slots_capacity"] /= 25245
            obs2["selling_price"] /= 2700
            obs2["lifespan"] /= 96

            obs2.drop(columns=["release_time", "life_expectancy", "cost_of_moving", "server_type", "action", "moved", "slots_size", "capacity"], inplace=True)
            
            obs2 = obs2.to_dict(orient='list')

            # Pad all entries with zeroes up to max_servers
            padding = [0.0] * (max_servers - len(self.fleet))
            for k in obs2.keys():
                obs[k] = np.array(obs2[k] + padding)

        # Fill in U, L, and P
        obs["utilization"] = np.array([self.ulp[0]])
        obs["normalised_lifespan"] = np.array([self.ulp[1]])
        obs["profit"] = np.array([self.ulp[2]])

        return obs
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self._init_vars()
        # need to include the first demand in the first observation
        return self._get_obs(), {}
    
    # Fix some parameters of the action space to accelerate learning
    # Fix the time steps
    # For buy, fill in the server id to any new id 
    # For move, fill in the server generation
    # For dismiss, fill in the datacenter id and server generation
    # For hold, nextstep must be true to forward to the next time step
    def map_action(self, action, timestep):
        action_map = ["buy", "move", "dismiss", "hold"]
        sgen_map = ["CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4", "GPU.S1", "GPU.S2", "GPU.S3"]
        mapped_action = {
            "time_step": timestep,
            "action": action_map[action[3]],
            "nextstep": int(action[4])
        }
        if action[3] == 0: # buy
            existing_sids = set(self.fleet['server_id']) if not self.fleet.empty else set()
            all_sids = set(range(1, max_servers+1))
            possible_new_sids = list(all_sids - existing_sids)
            random_new_sid = np.random.choice(possible_new_sids)

            mapped_action["datacenter_id"] = f"DC{action[0] + 1}"
            mapped_action["server_generation"] = sgen_map[action[1]]
            mapped_action["server_id"] = int(random_new_sid)
            return mapped_action
        
        elif action[3] == 1: # move
            mapped_action["datacenter_id"] = f"DC{action[0] + 1}"
            mapped_action["server_id"] = int(action[2] + 1)
            mapped_action["server_generation"] = self.fleet.loc[self.fleet['server_id'] == mapped_action["server_id"], 'server_generation'].values[0]
        
        elif action[3] == 2: # dismiss
            sid = int(action[2] + 1)
            mapped_action["datacenter_id"] = self.fleet.loc[self.fleet['server_id'] == sid, 'datacenter_id'].values[0]
            mapped_action["server_generation"] = self.fleet.loc[self.fleet['server_id'] == sid, 'server_generation'].values[0]
            mapped_action["server_id"] = sid

        else: # hold
            mapped_action["datacenter_id"] = f"DC{action[0] + 1}"
            mapped_action["server_generation"] = sgen_map[action[1]]
            mapped_action["server_id"] = int(action[2] + 1)

        return mapped_action

    
    def action_masks(self):
        # 1. Check if server id is valid ( buy needs to create a new id, move/dismiss needs existing id)
        for i in range(len(self.action_mask)):
            self.action_mask[i] = True

        # if available to buy at that time step, only do buy action, unless fleet is full (max_servers reached)
        # need to assign a new server id
        if self.time_step in self.rtimes and len(self.fleet) < max_servers:
            # If this is the first time step, force the model to buy at least 100 servers
            if self.time_step == 1 and self.num_consec_buys < 100:
                self.action_mask[-1] = False

            # Set move, dismiss and hold to false
            for i in [1, 2, 3]:
                self.action_mask[num_datacenters + num_server_gens + max_servers + i] = False
            # Set server id to one value (0), we will randomly assign one
            for sid in range(1, max_servers):
                self.action_mask[num_datacenters + num_server_gens + sid] = False
        elif not self.fleet.empty: # cannot buy at this timestep, just move/dismiss/hold
            # Set buy to false
            self.action_mask[num_datacenters + num_server_gens + max_servers] = False 
            # Only existing server ids can be chosen
            sid_set = set([sid for sid in self.fleet["server_id"]])
            for i in range(max_servers):
                if i+1 not in sid_set:
                    self.action_mask[num_datacenters + num_server_gens + i] = False
            # Set server generation to 1 value, we will assign it based on server id
            for i in range(1, num_server_gens):
                self.action_mask
        else: # Hold and go to next time step
            for i in [0, 1, 2]:
                self.action_mask[num_datacenters + num_server_gens + max_servers + i] = False
            # Must continue to next time step
            self.action_mask[-2] = False 
        
        return self.action_mask

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
            
            self.ulp = [U, L, P] 

            reward = U * L * P

            # PUT ENTIRE FLEET on HOLD ACTION
            self.fleet = put_fleet_on_hold(self.fleet)

        return reward

    def step(self, action):
        terminated = bool(self.time_step >= num_timesteps)
        truncated = False # No need to truncate here, already set max_episode_steps when registering

        # GET THE SERVERS DEPLOYED AT TIMESTEP ts
        # Check if constraints are obeyed
        mapped_action = self.map_action(action, self.time_step)
        if (mapped_action["action"] == "buy"):
            self.num_consec_buys += 1
        else: 
            self.num_consec_buys = 0
        # if not self.is_sgen_valid(mapped_action):
        #     if action[4] == 1:
        #         self.time_step += 1
        #     return (self._get_obs(), invalid_reward, terminated, truncated, {})
        try:
            solution = solution_data_preparation(pd.DataFrame(mapped_action, index=[0]), self.servers, self.datacenters, self.selling_prices)
            ts_fleet = get_time_step_fleet(solution, self.time_step)

            # UPDATE FLEET
            new_fleet = update_fleet(self.time_step, self.fleet, ts_fleet)

            check_datacenter_slots_size_constraint(new_fleet)
            self.fleet = new_fleet 
        except Exception as e:
            if action[4] == 1:
                self.time_step += 1
            return (self._get_obs(), invalid_reward, terminated, truncated, {"valid": False})

        reward = self.calculate_reward()

        # print(f"reward at time step {self.time_step} with action {mapped_action} is {reward}")

        if action[4] == 1:
            self.time_step += 1

        # Optionally we can pass additional info, we are not using that for now
        info = {"valid": True}

        return (self._get_obs(), reward, terminated, truncated, info)

    def render(self):
        pass
        # print("Timestep:", self.time_step)
        # print("State:", self._get_obs())
        # print("Fleet:", self.fleet)

    def close(self):
        pass