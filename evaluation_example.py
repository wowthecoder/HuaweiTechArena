

from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function
from seeds import known_seeds

# LOAD SOLUTION
best_solution = load_solution('./output/best_solution.json')
# best_solution = load_solution('./test_output/1741.json')
# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices = load_problem_data()

# EVALUATE THE SOLUTION
# training_seeds = known_seeds('training')
# bestscore = 0
# for seed in training_seeds:
#     solution = load_solution(f'./output/{seed}.json')
#     score = evaluation_function(solution,
#                             demand,
#                             datacenters,
#                             servers,
#                             selling_prices,
#                             seed=seed,
#                             verbose=True)
#     bestscore = max(bestscore, score)
#     print("Score on", seed, "is", score)

# print("Best score is", bestscore)

# Evaluate best solution
bestscore = evaluation_function(best_solution,
                            demand,
                            datacenters,
                            servers,
                            selling_prices,
                            seed=1741,
                            verbose=True)

print(f'Best solution score on seed 1061: {bestscore}')