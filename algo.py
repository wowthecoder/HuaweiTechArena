import json
import uuid
import numpy as np
import random
from evaluation import check_datacenter_slots_size_constraint, check_release_time, evaluation_function, get_time_step_fleet
from utils import load_problem_data, load_solution, save_solution
import pandas as pd

# Assuming we have an evaluate_solution function that takes in an action sequence
# and returns the corresponding objective score
def evaluate_solution(action_sequence, demand, datacenters, servers, selling_prices, seeds):
    # Placeholder: This function should be implemented using the provided evaluation script
    score = 0
    # for seed in seeds:
    #     
    score += evaluation_function(action_sequence, demand, datacenters, servers, selling_prices, seed=123)
    
    return score

# Generate an initial random solution (action sequence)
def generate_initial_solution():
    file_path = './output/best_solution.json'
    solution = load_solution(file_path)

    return solution

# Generate a neighbor by modifying the current action sequence
def generate_neighbor(current_sequence):
    while True:
        neighbor = current_sequence.copy()
        ts = random.randint(1, 168)
        datacenter = random.choice(['DC1', 'DC2', 'DC3', 'DC4'])
        server = random.choice(['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3'])
        action = random.choice(['buy', 'move', 'dismiss'])
        
        rt = eval(servers[servers['server_generation'] == server]['release_time'].sample().values[0])
        
        if ts < min(rt) or ts > max(rt):
            continue
        
        fleet = get_time_step_fleet(neighbor, ts)
        
        if fleet.empty:
            fleet = pd.DataFrame(columns=['datacenter_id', 'server_generation', 'slots_size'])  # Ensure required columns are present
        
        id = str(uuid.uuid4())+'1'
        if action == 'dismiss':
            # check if there are servers of that particular type in the datacenter at that time
            if fleet.empty or fleet[(fleet['datacenter_id'] == datacenter) & (fleet['server_generation'] == server)].empty:
                continue
            id = fleet[(fleet['datacenter_id'] == datacenter) & (fleet['server_generation'] == server)]['server_id'].sample().values[0]
        
        if action == 'move':
            # check if there are servers of that particular type at that time
            if fleet.empty or fleet[(fleet['server_generation'] == server)].empty:
                continue            
            id = fleet[fleet['server_generation'] == server]['server_id'].sample().values
            # randomly select one of the available values
            id = random.choice(id)
            
        # Add new record to neighbor
        new_record = pd.DataFrame([{
            'time_step': ts,
            'datacenter_id': datacenter,
            'server_generation': server,
            'server_id': id,
            'action': action
        }])
        new_neighbor = pd.concat([neighbor, new_record], ignore_index=True)

        try:
            evaluation = evaluate_solution(new_neighbor, demand, datacenters, servers, selling_prices, seeds=[123])
            print(new_record)
            return (new_neighbor, evaluation)
        except Exception as e:
            print("neighbour error:",e)
            continue


# Simulated Annealing implementation
def simulated_annealing(initial_solution, initial_temp, cooling_rate, stop_temp, iterations):
    current_solution = initial_solution
    current_temp = initial_temp
    best_solution = current_solution
    
    seeds = []
    # for i in range(3):
    #     seed = random.randint(0, 2**32 - 1)
    #     seeds.append(seed)
    
    best_cost = 409294846.91128045
    print(best_cost)
    while current_temp > stop_temp:
        for _ in range(iterations):
            neighbor_solution, neighbor_cost = generate_neighbor(current_solution)
            neighbor_cost = float(neighbor_cost)
            print(neighbor_cost)
            delta_cost = neighbor_cost - best_cost

            if delta_cost > 0 or random.random() < np.exp(-delta_cost / current_temp):
                current_solution = neighbor_solution
                if neighbor_cost > best_cost:
                    best_solution = current_solution
                    # save best_solution in a separate file
                    save_solution(best_solution, './output/best_solution.json')
                    best_cost = neighbor_cost

        current_temp *= cooling_rate

    return best_solution, best_cost

# Parameters for Simulated Annealing
demand, datacenters, servers, selling_prices = load_problem_data()
initial_solution = generate_initial_solution()
initial_temp = 1000
cooling_rate = 0.85
stop_temp = 1
iterations_per_temp = 20

# Run Simulated Annealing
best_solution, best_solution_cost = simulated_annealing(
    initial_solution, initial_temp, cooling_rate, stop_temp, iterations_per_temp
)

print("Best Solution:", best_solution)
print("Best Solution Cost:", best_solution_cost)
