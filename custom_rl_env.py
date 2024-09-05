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

# Fix the time steps
# Fix action to buy at timestep 1
def map_action(action, timestep):
    action_map = ["buy", "move", "dismiss"]
    sgen_map = ["CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4", "GPU.S1", "GPU.S2", "GPU.S3"]
    # Scale it back to the original discrete action space
    # Convert from [-1, 1] to [0, 1]
    scaled_action = (action + 1) / 2
    action = [
        round(scaled_action[0] * 3),  # Map to [0, 3] (4 possible values)
        round(scaled_action[1] * (num_server_gens - 1)),  # Map to [0, num_server_gens - 1]
        round(scaled_action[2] * (max_servers - 1) + 1),  # Map to [1, 20000] (20000 possible values)
        round(scaled_action[3] * 2),  # Map to [0, 2] (3 possible values)
        round(scaled_action[4])
    ]
    if timestep == 1:
        actual_action = "buy"
    else:
        actual_action = action_map[action[3]]
    return {
        "time_step": timestep,
        "datacenter_id": f"DC{action[0] + 1}",
        "server_generation": sgen_map[action[1]],
        "server_id": action[2],
        "action": actual_action,
        "nextstep": action[4]
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
        #     "datacenter_id": Discrete(4), # 0: DC1, 1: DC2, 2: DC3, 3: DC4
        #     "server_generation": Discrete(num_server_gens), # 0: CPU.S1, 1: CPU.S2, 2: CPU.S3, 3: CPU.S4, 4: GPU.S1, 5: GPU.S2, 6: GPU.S3
        #     "server_id": Discrete(28000), # max 28000 servers ( Total 55845 slots across all data centers, min 2 slots per server)
        #     "action": Discrete(4) # 0: buy, 1: move, 2: dismiss, 3: hold
        #     "continue": Discrete(2) # 0: stay in current time step, 1: next time step
        # }))
        # According to the Tips and Tricks page of Stable Baselines3 docs, it is recommended to rescale action space to [-1, 1]
        # we update the timestep ourselves, don't let the model learn
        self.action_space = Box(
            low=np.array([-1, -1, -1, -1, -1], dtype=np.float32), 
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32)
        )

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
            "time_step": Discrete(num_timesteps),
            # fleet state
            "datacenter_id": fleet_col,
            "server_generation": fleet_col,
            "server_id": fleet_col,
            "action": fleet_col,
            "server_type": fleet_col,
            "release_time1": fleet_col,
            "release_time2": fleet_col,
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
            "cost": Box(low=0, high=1e6, shape=(max_servers,), dtype=np.float64)
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
            obs["action"] = zeros
            obs["server_type"] = zeros
            obs["release_time1"] = zeros
            obs["release_time2"] = zeros
            obs["purchase_price"] = zeros
            obs["slots_size"] = zeros # either 2 or 4
            obs["energy_consumption"] = zeros
            obs["capacity"] = zeros
            obs["average_maintenance_fee"] = zeros
            obs["cost_of_energy"] = zeros
            obs["latency_sensitivity"] = zeros
            obs["slots_capacity"] = zeros
            obs["selling_price"] = zeros
            obs["lifespan"] = zeros
            obs["moved"] = zeros
            obs["cost"] = zeros
        else:
            # Convert all the values in Fleet to numerical
            dcid_map = {"DC1": 1, "DC2": 2, "DC3": 3, "DC4": 4}
            sgen_map = {"CPU.S1": 1, "CPU.S2": 2, "CPU.S3": 3, "CPU.S4": 4, "GPU.S1": 5, "GPU.S2": 6, "GPU.S3": 7}
            action_map = {"buy": 1, "move": 2, "dismiss": 3, "hold": 4}
            stype_map = {"CPU": 1, "GPU": 2}
            latency_map = {"low": 1, "medium": 2, "high": 3}
            obs2 = self.fleet.copy()
            # Map to numeric values and normalise all columns except cost (becuz we dk the max and min values)
            # also cost_of_energy is already within 0 and 1
            # moved is a boolean value so it is either 0 or 1
            obs2["datacenter_id"] = (obs2["datacenter_id"].map(dcid_map)) / 4
            obs2["server_generation"] = (obs2["server_generation"].map(sgen_map)) / num_server_gens
            obs2["server_id"] /= max_servers
            obs2["action"] = (obs2["action"].map(action_map)) / 4
            obs2["server_type"] = (obs2["server_type"].map(stype_map)) / 2
            obs2["latency_sensitivity"] = (obs2["latency_sensitivity"].map(latency_map)) / 3
            obs2[["release_time1", "release_time2"]] = obs2["release_time"].str.strip('[]').str.split(',', expand=True)
            obs2[["release_time1", "release_time2"]] = (obs2[["release_time1", "release_time2"]].astype(int)) / 168
            obs2["purchase_price"] /= 160000
            obs2["slots_size"] /= 4
            obs2["energy_consumption"] /= 4200
            obs2["capacity"] /= 160
            obs2["average_maintenance_fee"] /= 3080
            obs2["slots_capacity"] /= 25245
            obs2["selling_price"] /= 2700
            obs2["lifespan"] /= 96

            obs2.drop(columns=["release_time", "life_expectancy", "cost_of_moving"], inplace=True)
            
            obs2 = obs2.to_dict(orient='list')

            # Pad all entries with zeroes up to max_servers
            padding = [0.0] * (max_servers - len(self.fleet))
            for k in obs2.keys():
                obs[k] = np.array(obs2[k] + padding)

        return obs
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.time_step = 1
        self.num_actions = 0
        self.fleet = pd.DataFrame()
        # need to include the first demand in the first observation
        return self._get_obs(), {}
    
    def is_action_valid(self, action):
        # 1. Check if server id is valid ( buy needs to create a new id, move needs the id to not exist in destination datacenter, dismiss need to use an existing id )
        # 3. Check if server is available for purchase at that time step
        # 4. Check if server move is valid ( the source datacenter has the server and the destination datacenter has enough slots)
        # 5. If move/dismiss, check if server generation matches id

        # if fleet is empty, return true if action is buy
        if self.fleet.empty:
            return True

        # Combine all server ids
        # combined_server_ids = {server_id for dc in self.data_centers for server_id in dc["servers_dict"].keyszeros}
        dcid, sgen, sid, act = action["datacenter_id"], action["server_generation"], action["server_id"], action["action"]
        rtimes = list(map(int, self.servers.loc[self.servers["server_generation"] == sgen, 'release_time'].values[0].strip('[]').split(',')))
        # center = self.data_centers[dcid]
        # server_info = self.server_info[sgen]
        if (act == "buy" and sid in self.fleet["server_id"]) \
        or (act != "buy" and sid not in self.fleet["server_id"]) \
        or (act == "buy" and self.time_step not in rtimes) \
        or (act != "buy" and sgen != self.fleet.loc[self.fleet["server_id"] == sid, 'server_generation'].values[0]):
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
        truncated = bool(self.num_actions > max_num_actions)
        self.num_actions += 1

        # GET THE SERVERS DEPLOYED AT TIMESTEP ts
        # Check if constraints are obeyed
        mapped_action = map_action(action, self.time_step)
        if not self.is_action_valid(mapped_action):
            reward = -10.0
            if action[4] > 0:
                self.time_step += 1
            return (self._get_obs(), reward, terminated, truncated, {})
        try:
            solution = solution_data_preparation(pd.DataFrame(mapped_action, index=[0]), self.servers, self.datacenters, self.selling_prices)
            ts_fleet = get_time_step_fleet(solution, self.time_step)

            # if ts_fleet.empty and not self.fleet.empty:
            #     ts_fleet = self.fleet
            # elif ts_fleet.empty and self.fleet.empty:
            #     return 0

            # UPDATE FLEET
            new_fleet = update_fleet(self.time_step, self.fleet, ts_fleet)

            check_datacenter_slots_size_constraint(new_fleet)
            self.fleet = new_fleet 
        except ValueError as ve:
            reward = -10.0
            if action[4] > 0:
                self.time_step += 1
            return (self._get_obs(), reward, terminated, truncated, {})

        reward = self.calculate_reward()

        # print(f"reward at time step {self.time_step} with action {mapped_action} is {reward}")

        if action[4] > 0:
            self.time_step += 1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (self._get_obs(), reward, terminated, truncated, info)

    def render(self):
        pass
        # print("Timestep:", self.time_step)
        # print("State:", self._get_obs())
        # print("Fleet:", self.fleet)

    def close(self):
        pass