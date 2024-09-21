
import numpy as np
import pandas as pd
from evaluation import get_actual_demand
import os
import uuid

def build_purchase(time_step, datacenter_id, server_generation, server_id, action):
    return {
        "time_step": time_step,
        "datacenter_id": datacenter_id,
        "server_generation": server_generation,
        "server_id": server_id,
        "action": action
    }

def get_my_solution(d):
    purchases = []
    datacenter_capacities = [25245, 15300, 7020, 8280]
    cpu_s1_server_ids = []
    cpu_s1_locations = []
    gpu_s3_server_ids = []
    gpu_s3_locations = []
    
    # Buy 'CPU.S1' servers for datacenters 3 and 4
    for i in range(4):
        for _ in range(datacenter_capacities[i] // 2):
            server_id = str(uuid.uuid4())
            purchases.append(build_purchase(1, f'DC{i+1}', 'CPU.S1', server_id, 'buy'))
            cpu_s1_server_ids.append(server_id)
            cpu_s1_locations.append(i+1)
    
    # Buy 'GPU.S3' servers for datacenters 3 and 4
    for i in range(4):
        for _ in range(datacenter_capacities[i] // 4):
            server_id = str(uuid.uuid4())
            purchases.append(build_purchase(97, f'DC{i+1}', 'GPU.S3', server_id, 'buy'))
            gpu_s3_server_ids.append(server_id)
            gpu_s3_locations.append(i+1)
    
    return purchases


demand = pd.read_csv('./data/demand.csv')
import json
# GET THE DEMAND
actual_demand = get_actual_demand(demand)
solution = get_my_solution(actual_demand)

# Load the solution_example.json to get the pricing_strategy
with open('./data/solution_example.json', 'r') as example_file:
    example_data = json.load(example_file)
    pricing_strategy = example_data['pricing_strategy']

# Replace 'fleet' with 'solution' and keep 'pricing_strategy'
with open('./data/solution_example.json', 'w') as output_file:
    json.dump({'fleet': solution, 'pricing_strategy': pricing_strategy}, output_file)
