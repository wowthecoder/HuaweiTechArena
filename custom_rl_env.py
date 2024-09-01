import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Sequence, Box, Tuple, MultiDiscrete # Be careful, sequence represents a tuple of variable length
import pandas as pd
from evaluation import get_time_step_demand, get_time_step_fleet, update_fleet, get_capacity_by_server_generation_latency_sensitivity, \
    check_datacenter_slots_size_constraint, get_utilization, get_normalized_lifespan, get_profit, put_fleet_on_hold

num_server_gens = 7

def format_actions(actions):
    action_map = ["buy", "move", "dismiss"]
    sgen_map = ["CPU.S1", "CPU.S2", "CPU.S3", "CPU.S4", "GPU.S1", "GPU.S2", "GPU.S3"]
    formatted_actions = []
    for action in actions:
        if action["action"] != 3:
            formatted_actions.append({
                "time_step": action[0],
                "datacenter_id": f"DC{action[1] + 1}",
                "server_generation": sgen_map[action[2]],
                "server_id": action[3],
                "action": action_map[action[4]]
            })
    return formatted_actions

class ServerFleetEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def _init_state(self, datacenters, demands, servers, selling_prices):
        self.data_centers = []  # list of data centers
        self.time_step = 0
        for i in range(len(datacenters)):
            dc = {
                "remaining_slots": datacenters.loc[i, 'slots_capacity'],
                "energy_cost": datacenters.loc[i, 'cost_of_energy'],
                "latency_sensitivity": datacenters.loc[i, 'latency_sensitivity'],
                # Need to convert to tuple when returning observation
                "servers_dict": {} # key: server_id, value: [generation (digit from 0-6), operating_time]
            }
            self.data_centers.append(dc)

        # server info 
        # convert the release time from string to list of integers
        servers["release_time"] = servers["release_time"].apply(lambda x: list(map(int, x.strip('[]').split(','))))
        self.server_info = tuple(servers.to_dict(orient='records'))
        self.selling_prices = selling_prices
        demands["demands"] = demands[demands.columns[-7:]].apply(tuple, axis=1) # convert the demands to a tuple
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
        self.action_space = Sequence(MultiDiscrete([168, 4, num_server_gens, 28000, 4]))

        # The observation will be the state of the data centers
        server_data = Dict({
            "generation": Discrete(num_server_gens),
            "server_id": Discrete(28000),
            "operating_time": Discrete(96),
        })
        server_info = Dict({
            "generation": Discrete(num_server_gens),
            "release_time": Tuple([Discrete(168)] * 2),
            "selling_price": Box(low=0, high=3000, shape=(1,), dtype=float),
            "purchase_price": Box(low=1, high=200000, shape=(1,), dtype=float),
            "slots_size": Box(low=1, high=5, shape=(1,), dtype=int),
            "energy_consumption": Box(low=1, high=5000, shape=(1,), dtype=int),
            "capacity": Box(low=1, high=200, shape=(1,), dtype=int),
            "life_expectancy": Discrete(1), # all 96
            "cost_of_moving": Discrete(1), # all 1000
            "avg_maintenance_cost": Box(low=0, high=3100, shape=(1,), dtype=int),
        })
        center_data = Dict({
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

    def _get_obs(self):
        obs = {}
        demand_ts = self.demands.loc[self.demands["time_step"] == self.time_step]  
        demand_list = demand_ts[['time_step', 'latency_sensitivity', 'demands']].to_dict(orient='records')
        obs["demand"] = tuple(demand_list)
        obs["time_step"] = self.time_step
        obs["server_info"] = self.server_info   
        for i, center in enumerate(self.data_centers):
            obs_dict = center.copy()
            # Convert the server dictionary to a tuple
            obs_dict.pop("servers_dict")
            server_list = [{"server_id": key, "generation": value[0], "operating_time": value[1]} for key, value in center["servers_dict"].items()]
            obs_dict["servers"] = tuple(server_list)
            obs["DC"+str(i+1)] = obs_dict

        return obs
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.time_step = 1
        for center in self.data_centers:
            center["remaining_slots"] = 0
            center["servers_dict"] = {}
        return self._get_obs(), {}  # empty info dict
    
    def is_action_valid(self, actions):
        # 1. Check if server id is valid ( buy needs to create a new id, move and dismiss need to use an existing id )
        # 2. Check if data center has enough slots left
        # 3. Check if server is available for purchase at that time step
        # 4. Check if server move is valid ( the source datacenter has the server and the destination datacenter has enough slots)

        # Combine all server ids
        combined_server_ids = {server_id for dc in self.data_centers for server_id in dc["servers_dict"].keys()}
        for action in actions:
            dcid, sgen, sid, act = action[1], action[2], action[3], action[4]
            center = self.data_centers[dcid]
            server_info = self.server_info[sgen]
            if (act == 0 and sid in combined_server_ids) \
            or (act == 1 and sid not in combined_server_ids) \
            or (act == 2 and sid not in center["servers_dict"].keys()) \
            or action[0] != self.time_step \
            or center["remaining_slots"] < server_info["slots_size"] \
            or (act == 0 and self.time_step not in server_info["release_time"]) \
            or (act == 1 and sid in center["servers_dict"].keys()) \
            or (act == 1 and center["remaining_slots"] < server_info["slots_size"]):
                return False
        return True 

    def calculate_reward(self, actions):
        # GET THE ACTUAL DEMAND AT TIMESTEP ts
        D = get_time_step_demand(self.demands, self.time_step)

        # GET THE SERVERS DEPLOYED AT TIMESTEP ts
        formatted_actions = format_actions(actions)
        ts_fleet = get_time_step_fleet(pd.DataFrame(formatted_actions), self.time_step)

        if ts_fleet.empty and not self.fleet.empty:
            ts_fleet = self.fleet
        elif ts_fleet.empty and self.fleet.empty:
            return 0

        # UPDATE FLEET
        self.fleet = update_fleet(self.time_step, self.fleet, ts_fleet)
  
        # CHECK IF THE FLEET IS EMPTY
        if self.fleet.shape[0] > 0:
            # GET THE SERVERS CAPACITY AT TIMESTEP ts
            Zf = get_capacity_by_server_generation_latency_sensitivity(self.fleet)
    
            # CHECK CONSTRAINTS
            check_datacenter_slots_size_constraint(self.fleet)
    
            # EVALUATE THE OBJECTIVE FUNCTION AT TIMESTEP ts
            U = get_utilization(D, Zf)
    
            L = get_normalized_lifespan(self.fleet)
    
            P = get_profit(D, 
                           Zf, 
                           self.selling_prices,
                           self.fleet)
            reward = U * L * P

            # PUT ENTIRE FLEET on HOLD ACTION
            self.fleet = put_fleet_on_hold(self.fleet)
        
        return reward

    def step(self, actions):
        # Check if action is valid
        if not self.is_action_valid(actions):
            # Optionally, penalize the agent for selecting an invalid action
            reward = -10.0
            terminated = bool(self.time_step == 168)
            truncated = False
            self.time_step += 1
            return self._get_obs(), reward, terminated, truncated, {}

        terminated = bool(self.time_step == 168)
        truncated = False  # we do not limit the number of steps here

        # Apply all actions
        for action in actions:
            center = self.data_centers[action[1]]
            sgen, sid, act = action[2], action[3], action[4]
            if act == 0:
                server = [action["server_generation"], 0]
                center["servers_dict"][sid] = server
                center["remaining_slots"] -= self.server_info[sgen]["slots_size"]
            elif act == 1:
                # Find the server in the source data center and remove it
                server = None
                for src_center in self.data_centers:
                    if sid in src_center["servers_dict"].keys():
                        server = src_center["servers_dict"].pop(sid)
                        src_center["remaining_slots"] += self.server_info[sgen]["slots_size"]
                        break
                # Add server to destination data center
                center["servers_dict"][sid] = server
                center["remaining_slots"] -= self.server_info[sgen]["slots_size"]
            elif act == 2: # dismiss
                # Remove server id from set
                center["servers_dict"].pop(sid)

        # Increment operating time for all servers
        # Automatically dismiss any server that is past the life expectancy
        for center in self.data_centers:
            for server in center["servers_dict"].values():
                server[1] += 1
                if server[1] >= self.server_info[server[0]]["life_expectancy"]:
                    center["servers_dict"].pop(sid)
                    center["remaining_slots"] += self.server_info[server[0]]["slots_size"]

        reward = self.calculate_reward(actions)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        self.time_step += 1

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        print("Timestep:", self.time_step)
        print("State:", self._get_obs())
        print("Fleet:", self.fleet)

    def close(self):
        pass