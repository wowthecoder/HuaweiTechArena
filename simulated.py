import json
import uuid
import numpy as np
import random
from evaluation import check_datacenter_slots_size_constraint, check_release_time, evaluation_function, get_actual_demand, get_time_step_fleet
from seeds import known_seeds
from utils import load_problem_data, load_solution, save_solution
import pandas as pd
from scipy.special import expit
import cProfile
import pstats
import os 

PROFILING = False

class Algorithm:

    def __init__(self):
        self.demand, self.datacenters, self.servers, self.selling_prices, self.elasticity = load_problem_data()
        self.initial_temp = 1000
        self.cooling_rate = 0.80
        self.stop_temp = 100 # (1) only does 1 iteration with 851
        self.iterations_per_temp = 5
    
    # Assuming we have an evaluate_solution function that takes in an action sequence
    # and returns the corresponding objective score
    def evaluate_solution(self, fleet, pricing_strategy, demand, datacenters, servers, selling_prices, elasticity, seeds):
        # Placeholder: This function should be implemented using the provided evaluation script
        score = evaluation_function(fleet=fleet, pricing_strategy=pricing_strategy, demand=demand, datacenters=datacenters, servers=servers, selling_prices=selling_prices, elasticity=elasticity, seed=123, verbose=1)
        
        return score
    

    def generate_initial_solution(self, seed):
        file_path = f'./output/{seed}.json'
        
        # Check if file exists
        if os.path.exists(file_path):
            solution = load_solution(file_path)
        else:
            # Load from best_solution.json if the file doesn't exist
            solution = load_solution('./output/best_solution.json')

        return solution

    # Generate a neighbor by modifying the current action sequence
    def generate_neighbor(self,current_sequence, seed):
        i = 0
        while True:
            fleet = current_sequence[0]
            pricing_strategy = current_sequence[1]
            neighbor = fleet.copy()
            t = random.choices([1,2], weights=[0.1, 0.9],k=1)[0]
            if t == 1:
                # delete a random entry in neighbor
                if neighbor.empty:
                    continue
                new_neighbor = neighbor.drop(neighbor.sample().index)
            else:
                datacenter = random.choice(['DC1', 'DC2', 'DC3', 'DC4'])
                action = random.choices(['buy', 'move', 'dismiss'], weights=[0.4,0.3,0.3],k=1)[0]

                ts = random.randint(1, 168)
                server = random.choice(['CPU.S1', 'CPU.S2', 'CPU.S3','CPU.S4','GPU.S1', 'GPU.S2','GPU.S3'])
                if server == 'CPU.S1' and ts > 60:
                    continue
                if server == 'CPU.S2' and (ts > 96 or ts < 37):
                    continue
                if server == 'CPU.S3' and (ts < 73 or ts > 132):
                    continue
                if server == 'CPU.S4' and ts < 109:
                    continue
                if server == 'GPU.S1' and ts > 72:
                    continue
                if server == 'GPU.S2' and (ts<49 or ts>125):
                    continue
                if server == 'GPU.S3' and ts < 97:
                    continue

                if (ts == 1 and action == 'move'):
                    continue
                rt = eval(self.servers[self.servers['server_generation'] == server]['release_time'].sample().values[0])
        
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
                evaluation = self.evaluate_solution(new_neighbor, pricing_strategy, self.demand, self.datacenters, self.servers, self.selling_prices, self.elasticity, seed)
                if t != 1:
                    print(new_record)
                return (new_neighbor, evaluation)
            except Exception as e:
                print("neighbour error:",e)
                continue


    # Simulated Annealing implementation
    def simulated_annealing(self, initial_solution, initial_temp, cooling_rate, stop_temp, iterations, seed):
        current_solution = initial_solution
        pricing_solution = initial_solution[1]
        current_temp = initial_temp
        best_solution = current_solution
        best_cost = 0
        print(best_cost)
        while current_temp > stop_temp:
            print(f"Current Temp: {current_temp}")
            for _ in range(iterations):
                neighbor_solution, neighbor_cost = self.generate_neighbor(current_solution, seed)
                if neighbor_cost is None:
                    neighbor_cost = 0
                neighbor_cost = float(neighbor_cost)
                print(neighbor_cost)
                delta_cost = neighbor_cost - best_cost

                if delta_cost > 0 or random.random() < expit(-delta_cost / current_temp):
                    current_solution = (neighbor_solution, pricing_solution)
                    if neighbor_cost > best_cost:
                        best_solution = current_solution
                        # save best_solution in a separate file
                        save_solution(best_solution[0], pricing_solution, './output/best_solution.json')
                        best_cost = neighbor_cost

            current_temp *= cooling_rate

        return best_solution, best_cost

    def generate_solution(self, seed):
        # Parameters for Simulated Annealing
        initial_solution = self.generate_initial_solution(seed)
        
        # Run Simulated Annealing
        best_solution, best_solution_cost = self.simulated_annealing(
            initial_solution, self.initial_temp, self.cooling_rate, self.stop_temp, self.iterations_per_temp, seed
        )

        return best_solution, best_solution_cost

if __name__ == '__main__':
    # Load global data
    seeds = known_seeds()
    demand = pd.read_csv('./data/demand.csv')
    # Train for each seed
    for seed in seeds:
        np.random.seed(seed)

        algo = Algorithm()

        if PROFILING:
            # Start the profiler
            profiler = cProfile.Profile()
            profiler.enable()
        
        solution, solution_cost = algo.generate_solution(seed)
        print(f'Solution cost for {seed}: {solution_cost}')

        if PROFILING:
            # Stop the profiler
            profiler.disable()
            # Save the profiling data to a file
            profiler.dump_stats(f'profile_data_{seed}.prof')
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(100)

        # Save the solution
        output_dir = './output/'
        os.makedirs(output_dir, exist_ok=True)
        save_solution(solution[0], solution[1], f'./output/{seed}.json')