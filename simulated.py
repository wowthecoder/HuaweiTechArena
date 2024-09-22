import json
import uuid
import numpy as np
import random
from evaluation import check_datacenter_slots_size_constraint, check_release_time, evaluation_function, get_actual_demand, get_new_demand_for_new_price, get_time_step_fleet
from seeds import known_seeds
from utils import load_problem_data, load_solution, save_solution
import pandas as pd
from scipy.special import expit
import cProfile
import pstats
import os 
from scipy.optimize import minimize

PROFILING = False

class Algorithm:

    def __init__(self):
        self.demand, self.datacenters, self.servers, self.selling_prices, self.elasticity = load_problem_data()
        self.initial_temp = 1000
        self.cooling_rate = 0.80
        self.stop_temp = 100 # (1) only does 1 iteration with 851
        self.iterations_per_temp = 5
        self.constraint = 1100
        self.new_constraint = 1100
    
    def evaluate_solution(self, fleet, pricing_strategy, demand, datacenters, servers, selling_prices, elasticity, seed):
        score = evaluation_function(fleet=fleet, pricing_strategy=pricing_strategy, demand=demand, datacenters=datacenters, servers=servers, selling_prices=selling_prices, elasticity=elasticity, seed=seed, verbose=1)
        return score

    def generate_initial_solution(self, seed):
        file_path = f'./output/{seed}.json'
        if os.path.exists(file_path):
            solution = load_solution(file_path)
        else:
            solution = load_solution('./output/best_solution.json')
        return solution
    def get_objective(self,delta: np.ndarray, base_selling_price: float, demands: np.ndarray, price_elasticity_of_demand: float, constraint) -> float:
        selling_prices = base_selling_price + delta
        max_demand = np.max(demands)
        new_demands = np.array([min(get_new_demand_for_new_price(d, base_selling_price, sp, price_elasticity_of_demand), constraint) 
                            for d, sp in zip(demands, selling_prices)])
        valid_indices = new_demands != 0
        sum = np.sum(new_demands[valid_indices] * selling_prices[valid_indices])
        return sum

    def neg_get_objective(self,delta: np.ndarray, base_selling_price: float, demands: np.ndarray, price_elasticity_of_demand: float, constraint) -> float:
        return -self.get_objective(delta, base_selling_price, demands, price_elasticity_of_demand, constraint)

    def optimize_please(self, name: str, sensitivity: str, constraint) -> np.ndarray:
        i = {'high': 2, 'medium': 1}.get(sensitivity, 0)
    
        base_selling_price = self.selling_prices[self.selling_prices['server_generation'] == name]['selling_price'].values[i]
        price_elasticity_of_demand = self.elasticity[self.elasticity['server_generation'] == name]['elasticity'].values[i]

    
        demands = self.demand[self.demand['latency_sensitivity'] == 'high'][name].values

        initial_delta = np.array([0.0] * 168)
        bounds = [(None, None)] * 168

        result = minimize(self.neg_get_objective, initial_delta, args=(base_selling_price, demands, price_elasticity_of_demand, constraint), bounds=bounds, method='Nelder-Mead')
        return result.x
    def optimize_price(self,constraint):
        delta = []
        for sensitivity in ['high', 'medium', 'low']:
            for name in ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']:
                cur = self.optimize_please(name, sensitivity, constraint).tolist()
                for i in cur:
                    delta.append(i+ self.selling_prices[(self.selling_prices['server_generation'] == name) & (self.selling_prices['latency_sensitivity'] == sensitivity)]['selling_price'].values[0])
        return delta
    def generate_neighbor(self, current_sequence, seed):
        i = 0
        while True:
            fleet = current_sequence[0]
            pricing_strategy = current_sequence[1]
            neighbor = fleet.copy()
            t = random.choices([1, 2, 3], weights=[0.1,0.9,0], k=1)[0]
            if t == 1:
                if neighbor.empty:
                    continue
                new_neighbor = neighbor.drop(neighbor.sample().index)
            elif t == 2:
                datacenter = random.choice(['DC1', 'DC2', 'DC3', 'DC4'])
                action = random.choice(['buy', 'move', 'dismiss'])
                ts = random.randint(1, 168)
                server = random.choice(['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3'])
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
                if server == 'GPU.S2' and (ts < 49 or ts > 125):
                    continue
                if server == 'GPU.S3' and ts < 97:
                    continue
                if action == 'move':
                    if ts == 1:
                        continue
                rt = eval(self.servers[self.servers['server_generation'] == server]['release_time'].sample().values[0])
                if ts < min(rt) or ts > max(rt):
                    continue
                fleet = get_time_step_fleet(neighbor, ts)
                if fleet.empty:
                    fleet = pd.DataFrame(columns=['datacenter_id', 'server_generation', 'slots_size'])
                id = str(uuid.uuid4()) + '1'
                if action == 'dismiss':
                    if fleet.empty or fleet[(fleet['datacenter_id'] == datacenter) & (fleet['server_generation'] == server)].empty:
                        continue
                    id = fleet[(fleet['datacenter_id'] == datacenter) & (fleet['server_generation'] == server)]['server_id'].sample().values[0]
                if action == 'move':
                    if fleet.empty or fleet[(fleet['server_generation'] == server)].empty:
                        continue
                    id = fleet[fleet['server_generation'] == server]['server_id'].sample().values
                    id = random.choice(id)
                new_record = pd.DataFrame([{
                    'time_step': ts,
                    'datacenter_id': datacenter,
                    'server_generation': server,
                    'server_id': id,
                    'action': action
                }])
                new_neighbor = pd.concat([neighbor, new_record], ignore_index=True)
            else:
                # Change pricing strategy
                new_pricing_strategy = pricing_strategy.copy()
                c = random.choice([-1,1])
                self.new_constraint = self.constraint + 10*c
                delta = self.optimize_price(self.new_constraint)
                new_pricing_strategy['price'] = delta
                new_neighbor = neighbor
                evaluation = self.evaluate_solution(new_neighbor, new_pricing_strategy, self.demand, self.datacenters, self.servers, self.selling_prices, self.elasticity, seed)
                return (new_neighbor, pricing_strategy, evaluation)
                

            try:
                evaluation = self.evaluate_solution(new_neighbor, pricing_strategy, self.demand, self.datacenters, self.servers, self.selling_prices, self.elasticity, seed)
                if t != 1:
                    print(new_record)
                return (new_neighbor, pricing_strategy, evaluation)
            except Exception as e:
                print("neighbour error:", e)
                continue

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
                neighbor_solution, neighbor_pricing, neighbor_cost = self.generate_neighbor(current_solution, seed)
                if neighbor_cost is None:
                    neighbor_cost = 0
                neighbor_cost = float(neighbor_cost)
                print(neighbor_cost)
                delta_cost = neighbor_cost - best_cost
                    
                if delta_cost > 0 or random.random() < expit(-delta_cost / current_temp):
                    current_solution = (neighbor_solution, neighbor_pricing)
                    self.constraint = self.new_constraint
                    if neighbor_cost > best_cost:
                        best_solution = current_solution
                        save_solution(best_solution[0], best_solution[1], './output/best_solution.json')
                        best_cost = neighbor_cost
                    self.constraint = self.new_constraint
                else:
                    self.new_constraint = self.constraint
            current_temp *= cooling_rate
        return best_solution, best_cost

    def generate_solution(self, seed):
        initial_solution = self.generate_initial_solution(seed)
        best_solution, best_solution_cost = self.simulated_annealing(
            initial_solution, self.initial_temp, self.cooling_rate, self.stop_temp, self.iterations_per_temp, seed
        )
        return best_solution, best_solution_cost

if __name__ == '__main__':
    seeds = known_seeds()
    demand = pd.read_csv('./data/demand.csv')
    for seed in seeds:
        np.random.seed(seed)
        algo = Algorithm()
        if PROFILING:
            profiler = cProfile.Profile()
            profiler.enable()
        solution, solution_cost = algo.generate_solution(seed)
        print(f'Solution cost for {seed}: {solution_cost}')
        if PROFILING:
            profiler.disable()
            profiler.dump_stats(f'profile_data_{seed}.prof')
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(100)
        output_dir = './output/'
        os.makedirs(output_dir, exist_ok=True)
        save_solution(solution[0], solution[1], f'./output/{seed}.json')