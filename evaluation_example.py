

from utils import (load_problem_data,
                   load_solution, save_solution)
from evaluation import evaluation_function
import json


# LOAD SOLUTION
fleet, pricing_strategy = load_solution('./output/123.json')
print(pricing_strategy.columns)
print(fleet)
# add 10 CPU.S1 to DC4 at timestep 1
# for i in range(10):
#     fleet = fleet._append({'time_step': 1, 'datacenter_id': 'DC4', 'server_generation': 'CPU.S1', 'server_id': str(i), 'action': 'buy'}, ignore_index=True)
fleet.drop_duplicates(inplace=True)
# fleet = fleet.sort_values(by=['time_step', 'server_generation', 'datacenter_id'], inplace=False)
# # fleet.reset_index(drop=True, inplace=True)
# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

sum = 0

# EVALUATE THE SOLUTION
for seed in [2381, 5351, 6047, 6829, 9221, 9859, 8053, 1097, 8677, 2521]:
    # save_solution(fleet, pricing_strategy, f'./output/{seed}.json')
    # if seed == 2381:
    #     continue
    score = evaluation_function(fleet=fleet, pricing_strategy=pricing_strategy, demand=demand, datacenters=datacenters, servers=servers, selling_prices=selling_prices, elasticity=elasticity, seed=seed, verbose=0)
    print(score)
    # sum+= score
    break
print(sum)




