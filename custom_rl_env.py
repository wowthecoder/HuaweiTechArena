import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Sequence, Box, Tuple, MultiDiscrete # Be careful, sequence represents a tuple of variable length
import pandas as pd
from evaluation import get_time_step_demand, get_time_step_fleet, update_fleet, get_capacity_by_server_generation_latency_sensitivity, \
    check_datacenter_slots_size_constraint, get_utilization, get_normalized_lifespan, get_profit, put_fleet_on_hold, \
    change_selling_prices_format, solution_data_preparation

num_server_gens = 7

def map_actions(actions):
    action_map = ["buy", "move", "dismiss"]
    sgen_map = ["CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4", "GPU.S1", "GPU.S2", "GPU.S3"]
    mapped_actions = []
    for action in actions:
        if action[4] != 3: # If the action is not "hold"
            mapped_actions.append({
                "time_step": action[0],
                "datacenter_id": f"DC{action[1]}",
                "server_generation": sgen_map[action[2]],
                "server_id": action[3],
                "action": action_map[action[4]]
            })
    return mapped_actions

class ServerFleetEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def _init_state(self, datacenters, demands, servers, selling_prices):
        # self.data_centers = []  # list of data centers
        self.server_ids = set()
        self.datacenters = datacenters
        self.time_step = 0
        # for i in range(len(datacenters)):
        #     dc = {
        #         "remaining_slots": datacenters.loc[i, 'slots_capacity'],
        #         "energy_cost": datacenters.loc[i, 'cost_of_energy'],
        #         "latency_sensitivity": datacenters.loc[i, 'latency_sensitivity'],
        #         # Need to convert to tuple when returning observation
        #         "servers_dict": {} # key: server_id, value: [generation (digit from 0-6), operating_time, selling_price]
        #     }
        #     self.data_centers.append(dc)

        # server info 
        # convert the release time from string to list of integers
        self.servers = servers
        # servers["release_time"] = servers["release_time"].apply(lambda x: list(map(int, x.strip('[]').split(','))))
        # servers.rename(columns={"server_generation": "generation"}, inplace=True)
        # servers.drop(columns=["server_type"], inplace=True)
        # self.server_info = tuple(servers.to_dict(orient='records'))

        # Ordinal encode the selling prices dataframe
        # selling_prices['sgen_encoded'] = pd.factorize(selling_prices['server_generation'])[0]
        # selling_prices['latency_encoded'] = pd.factorize(selling_prices['latency_sensitivity'])[0]
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
        # Hold action is included because a Sequence must return at least one action, but sometimes we want to do nothing at a time step
        # Dictionary action space is not supported, but leaving the commented code here for description 
        # self.action_space = Sequence(Dict({
        #     "time_step": Discrete(168), 
        #     "datacenter_id": Discrete(4), # 0: DC1, 1: DC2, 2: DC3, 3: DC4
        #     "server_generation": Discrete(num_server_gens), # 0: CPU.S1, 1: CPU.S2, 2: CPU.S3, 3: CPU.S4, 4: GPU.S1, 5: GPU.S2, 6: GPU.S3
        #     "server_id": Discrete(28000), # max 28000 servers ( Total 55845 slots across all data centers, min 2 slots per server)
        #     "action": Discrete(4) # 0: buy, 1: move, 2: dismiss, 3: hold
        # }))
        self.action_space = Sequence(MultiDiscrete([168, 4, num_server_gens, 28000, 4], start=[1, 1, 0, 1, 0]))

        # The observation will be the state of the data centers
        # server_data = Dict({
        #     "generation": Discrete(num_server_gens),
        #     "server_id": Discrete(28000),
        #     "operating_time": Discrete(96),
        #     "selling_price": Box(low=0, high=3000, shape=(1,), dtype=float),
        # })
        # server_info = Dict({
        #     "generation": Discrete(num_server_gens),
        #     "release_time": Tuple([Discrete(168)] * 2),
        #     "purchase_price": Box(low=1, high=200000, shape=(1,), dtype=float),
        #     "slots_size": Box(low=1, high=5, shape=(1,), dtype=int),
        #     "energy_consumption": Box(low=1, high=5000, shape=(1,), dtype=int),
        #     "capacity": Box(low=1, high=200, shape=(1,), dtype=int),
        #     "life_expectancy": Discrete(1), # all 96
        #     "cost_of_moving": Discrete(1), # all 1000
        #     "average_maintenance_fee": Box(low=0, high=3100, shape=(1,), dtype=int),
        # })
        # center_data = Dict({
        #     "remaining_slots": Box(low=0, high=26000, shape=(1,), dtype=int),
        #     "energy_cost": Box(low=0.0, high=1.0, shape=(1,), dtype=float),
        #     "latency_sensitivity": Discrete(3), # 0: low, 1: medium, 2: high
        #     "servers": Sequence(server_data)
        # })
        demand_data = Dict({
            "server_generation": Discrete(7),
            "high": Box(low=0, high=1e6, shape=(1,), dtype=int),
            "medium": Box(low=0, high=1e6, shape=(1,), dtype=int),
            "low": Box(low=0, high=1e6, shape=(1,), dtype=int),
        })
        # self.observation_space = Dict({
        #     "demands": Sequence(demand_data),
        #     "time_step": Discrete(168),
        #     "server_info": Tuple([server_info] * num_server_gens),
        #     "DC1": center_data,
        #     "DC2": center_data,
        #     "DC3": center_data,
        #     "DC4": center_data
        # })

        self.observation_space = Dict({
            "demands": Sequence(demand_data),
            "time_step": Discrete(168, start=1),
            "datacenter_id": Sequence(Discrete(4, start=1)),
            "server_generation": Sequence(Discrete(7)),
            "server_id": Sequence(Discrete(28000)),
            "action": Sequence(Discrete(4)),
            "server_type": Sequence(Discrete(2)),
            "release_time_1": Sequence(Discrete(168)),
            "release_time_2": Sequence(Discrete(168)),
            "purchase_price": Sequence(Discrete(2e5)),
            "slots_size": Sequence(Discrete(5)), # either 2 or 4
            "energy_consumption": Sequence(Discrete(5000)),
            "capacity": Sequence(Discrete(160)),
            "life_expectancy": Discrete(1, start=96),
            "cost_of_moving": Discrete(1, start=1000),
            "average_maintenance_fee": Sequence(Discrete(3100)),
            "cost_of_energy": Sequence(Box(low=0.0, high=1.0, shape=(1,), dtype=float)),
            "latency_sensitivity": Sequence(Discrete(3)),
            "slots_capacity": Sequence(Discrete(26000)),
            "selling_price": Sequence(Discrete(3000)),
            "lifespan": Sequence(Discrete(100)),
            "moved": Sequence(Discrete(2)),
            "cost": Sequence(Box(low=0, high=1e5, shape=(1,), dtype=float))
        })

    def _get_obs(self):
        obs = {}
        # for i, center in enumerate(self.data_centers):
        #     obs_dict = center.copy()
        #     # Convert the server dictionary to a tuple
        #     obs_dict.pop("servers_dict")
        #     server_list = [{"server_id": key, "generation": value[0], "operating_time": value[1], "selling_price": value[2]} \
        #         for key, value in center["servers_dict"].items()]
        #     obs_dict["servers"] = tuple(server_list)
        #     obs["DC"+str(i+1)] = obs_dict

        # Convert all the values in Fleet to numerical
        dcid_map = {"DC1": 1, "DC2": 2, "DC3": 3, "DC4": 4}
        sgen_map = {"CPU.S1": 0, "CPU.S2": 1, "CPU.S3": 2, "CPU.S4": 3, "GPU.S1": 4, "GPU.S2": 5, "GPU.S3": 6}
        action_map = {"buy": 0, "move": 1, "dismiss": 2, "hold": 4}
        stype_map = {"CPU": 0, "GPU": 1}
        latency_map = {"low": 0, "medium": 1, "high": 2}
        obs = self.fleet.to_dict(orient='list')
        obs["datacenter_id"] = obs["datacenter_id"].map(dcid_map)
        obs["server_generation"] = obs["server_generation"].map(sgen_map)
        obs["action"] = obs["action"].map(action_map)
        obs["server_type"] = obs["server_type"].map(stype_map)
        obs["latency_sensitivity"] = obs["latency_sensitivity"].map(latency_map)
        obs[["release_time1", "release_time2"]] = obs["release_time"].str.strip('[]').str.split(',', expand=True)
        obs.drop(columns="release_time", inplace=True)

        demand_ts = self.demands.loc[self.demands["time_step"] == self.time_step]  
        demand_list = demand_ts.drop(columns="time_step", inplace=False).to_dict(orient='list')
        obs["demands"] = tuple(demand_list)
        obs["time_step"] = self.time_step 

        return obs
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.time_step = 0
        self.fleet = pd.DataFrame()
        # for center in self.data_centers:
        #     center["remaining_slots"] = 0
        #     center["servers_dict"] = {}
        return self._get_obs(), {}  # empty info dict
    
    def is_action_valid(self, actions):
        # 1. Check if server id is valid ( buy needs to create a new id, move needs the id to not exist in destination datacenter, dismiss need to use an existing id )
        # 3. Check if server is available for purchase at that time step
        # 4. Check if server move is valid ( the source datacenter has the server and the destination datacenter has enough slots)
        # 5. If move/dismiss, check if server generation matches id

        # Combine all server ids
        # combined_server_ids = {server_id for dc in self.data_centers for server_id in dc["servers_dict"].keys()}
        sgen_map = ["CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4", "GPU.S1", "GPU.S2", "GPU.S3"]
        for action in actions:
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

    def step(self, actions):
        self.time_step += 1
        # Check if action is valid
        if not self.is_action_valid(actions):
            # Optionally, penalize the agent for selecting an invalid action
            reward = -10.0
            terminated = bool(self.time_step == 168)
            truncated = False
            return self._get_obs(), reward, terminated, truncated, {}

        terminated = bool(self.time_step == 168)
        truncated = False  # we do not limit the number of steps here

        # Apply all actions ( the commented code is my manual approach )
        # for action in actions:
        #     center = self.data_centers[action[1]]
        #     sgen, sid, act = action[2], action[3], action[4]
        #     prices = self.selling_prices
        #     if act == 0:
        #         selling_price = prices.loc[prices["sgen_encoded"] == sgen & prices["latency_encoded"] == center["latency_sensitivity"], "selling_price"].iloc[0]
        #         server = [sgen, 0, selling_price]
        #         center["servers_dict"][sid] = server
        #         center["remaining_slots"] -= self.server_info[sgen]["slots_size"]
        #     elif act == 1:
        #         # Find the server in the source data center and remove it
        #         server = None
        #         for src_center in self.data_centers:
        #             if sid in src_center["servers_dict"].keys():
        #                 server = src_center["servers_dict"].pop(sid)
        #                 src_center["remaining_slots"] += self.server_info[sgen]["slots_size"]
        #                 break
        #         selling_price = prices.loc[prices["sgen_encoded"] == sgen & prices["latency_encoded"] == center["latency_sensitivity"], "selling_price"].iloc[0]
        #         server[2] = selling_price
        #         # Add server to destination data center
        #         center["servers_dict"][sid] = server
        #         center["remaining_slots"] -= self.server_info[sgen]["slots_size"]
        #     elif act == 2: # dismiss
        #         # Remove server id from set
        #         center["servers_dict"].pop(sid)

        # Increment operating time for all servers
        # Automatically dismiss any server that is past the life expectancy
        # for center in self.data_centers:
        #     for server in center["servers_dict"].values():
        #         server[1] += 1
        #         if server[1] >= self.server_info[server[0]]["life_expectancy"]:
        #             center["servers_dict"].pop(sid)
        #             center["remaining_slots"] += self.server_info[server[0]]["slots_size"]

        # GET THE SERVERS DEPLOYED AT TIMESTEP ts
        mapped_actions = map_actions(actions)
        solution = solution_data_preparation(pd.DataFrame(mapped_actions), self.servers, self.datacenters, self.selling_prices)
        ts_fleet = get_time_step_fleet(solution, self.time_step)

        if ts_fleet.empty and not self.fleet.empty:
            ts_fleet = self.fleet
        elif ts_fleet.empty and self.fleet.empty:
            return 0

        # UPDATE FLEET
        new_fleet = update_fleet(self.time_step, self.fleet, ts_fleet)

        # Check if slot constraint is obeyed
        try: 
            check_datacenter_slots_size_constraint(new_fleet)
            self.fleet = new_fleet 
        except ValueError as ve:
            reward = -10.0
            terminated = bool(self.time_step == 168)
            truncated = False
            return self._get_obs(), reward, terminated, truncated, {}

        reward = self.calculate_reward()

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (self._get_obs(), reward, terminated, truncated, info)

    def render(self):
        print("Timestep:", self.time_step)
        print("State:", self._get_obs())
        print("Fleet:", self.fleet)

    def close(self):
        pass