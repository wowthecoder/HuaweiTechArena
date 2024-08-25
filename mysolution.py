
import json
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
import os
import uuid

def get_my_solution(d):
    purchases = []
    for _ in range(420):
        purchase = {
            "time_step": 1,
            "datacenter_id": "DC1",
            "server_generation": "CPU.S1",
            "server_id": str(uuid.uuid4()),  # Generates a unique UUID
            "action": "buy"
        }
        purchases.append(purchase)
    for _ in range(255):
        purchase = {
            "time_step": 1,
            "datacenter_id": "DC2",
            "server_generation": "CPU.S1",
            "server_id": str(uuid.uuid4()),  # Generates a unique UUID
            "action": "buy"
        }
        purchases.append(purchase)
    for _ in range(117):
        purchase = {
            "time_step": 1,
            "datacenter_id": "DC3",
            "server_generation": "CPU.S1",
            "server_id": str(uuid.uuid4()),  # Generates a unique UUID
            "action": "buy"
        }
        purchases.append(purchase)
    
    for _ in range(138):
        purchase = {
            "time_step": 1,
            "datacenter_id": "DC4",
            "server_generation": "CPU.S1",
            "server_id": str(uuid.uuid4()),  # Generates a unique UUID
            "action": "buy"
        }
        purchases.append(purchase)

    for _ in range(105):
        purchase = {
            "time_step": 97,
            "datacenter_id": "DC1",
            "server_generation": "CPU.S3",
            "server_id": str(uuid.uuid4()),  # Generates a unique UUID
            "action": "buy"
        }
        purchases.append(purchase)
    
    for _ in range(79):
        purchase = {
            "time_step": 97,
            "datacenter_id": "DC1",
            "server_generation": "CPU.S4",
            "server_id": str(uuid.uuid4()),  # Generates a unique UUID
            "action": "buy"
        }
        purchases.append(purchase)
    
    for _ in range(127):
        purchase = {
            "time_step": 97,
            "datacenter_id": "DC2",
            "server_generation": "CPU.S3",
            "server_id": str(uuid.uuid4()),  # Generates a unique UUID
            "action": "buy"
        }
        purchases.append(purchase)
    
    for _ in range(58):
        purchase = {
            "time_step": 97,
            "datacenter_id": "DC3",
            "server_generation": "CPU.S3",
            "server_id": str(uuid.uuid4()),  # Generates a unique UUID
            "action": "buy"
        }
        purchases.append(purchase)
    
    for _ in range(69):
        purchase = {
            "time_step": 97,
            "datacenter_id": "DC4",
            "server_generation": "CPU.S3",
            "server_id": str(uuid.uuid4()),  # Generates a unique UUID
            "action": "buy"
        }
        purchases.append(purchase)
    return purchases

seeds = known_seeds('training')

demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)
    file_path = './output/best_solution.json'
    solution = []
    # Open and load the JSON file
    with open(file_path, 'r') as file:
        solution = json.load(file)
    output_dir = './output/'
    os.makedirs(output_dir, exist_ok=True)
    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')

