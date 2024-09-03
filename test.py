import pandas as pd
from gymnasium.spaces import Box
import numpy as np

servers = pd.read_csv('./data/servers.csv')
fleet = pd.read_csv('./rl_data/fleet.csv')
demands = pd.read_csv('./rl_data/actual_demand_1061.csv')

# demand_ts = demands.loc[demands["time_step"] == 2] 
# demand_list = demand_ts.drop(columns="time_step", inplace=False).to_dict(orient='list')
# print(demand_list)
# print(type(demand_list))

test_id = '71f734a2-32b2-46df-87f7-adceed8ae310'
sgen = fleet.loc[fleet["server_id"] == test_id, 'server_generation'].values[0]

print(sgen)

# Count the frequency of each unique time_step
# time_step_counts = demands['time_step'].value_counts()

# Find the maximum frequency
# max_frequency_value = time_step_counts.idxmax()
# max_frequency = time_step_counts.max()

# print(f"The time_step value with the maximum frequency is: {max_frequency_value} (Frequency: {max_frequency})")

test = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
sample = test.sample()
print(sample)
print(type(sample))

['high', 'low', 'medium', 'demand_generation', 'time_step', 'datacenter_id', 'server_generation', 'server_id', 'action', 'server_type', 'release_time_1', 'release_time_2', 'purchase_price', 'slots_size', 'energy_consumption', 'capacity', 'average_maintenance_fee', 'cost_of_energy', 'latency_sensitivity', 'slots_capacity', 'selling_price', 'lifespan', 'moved']
['action', 'average_maintenance_fee', 'capacity', 'cost', 'cost_of_energy', 'datacenter_id', 'demand_generation', 'energy_consumption', 'high', 'latency_sensitivity', 'lifespan', 'low', 'medium', 'moved', 'purchase_price', 'release_time_1', 'release_time_2', 'selling_price', 'server_generation', 'server_id', 'server_type', 'slots_capacity', 'slots_size', 'time_step']