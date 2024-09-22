import numpy as np
import pandas as pd
from uuid import uuid1
from seeds import known_seeds
from utils import load_problem_data, load_solution, save_solution
from evaluation import (
    change_elasticity_format,
    change_selling_prices_format,
    fleet_data_preparation,
    get_actual_demand,
    get_capacity_by_server_generation_latency_sensitivity,
    get_profit,
    get_time_step_demand,
    get_time_step_fleet,
    get_time_step_prices,
    pricing_data_preparation,
    update_fleet,
    update_selling_prices,
    update_demand_according_to_prices
)

def get_my_solution(d, initial_fleet, constant_pricing_strategy):
    fleet_filename = 'output/fleet.csv'
    cols = ["time_step", "datacenter_id", "server_generation", "server_id", "action", "server_type", "release_time", "purchase_price", "slots_size", "energy_consumption", "capacity", "life_expectancy", "cost_of_moving", "average_maintenance_fee", "cost_of_energy", "latency_sensitivity", "slots_capacity", "selling_price", "lifespan", "moved", "cost"]
    FLEET = pd.read_csv(fleet_filename)
    FLEET.columns = cols
    print(FLEET)
    PRICING_STRAT = constant_pricing_strategy
    PRICING_STRAT.columns = ['time_step', 'latency_sensitivity', 'server_generation', 'price']
    
    # Clean lifespan and life_expectancy columns to remove non-numeric characters
    numerical = ['purchase_price', 'slots_size', 'energy_consumption', 'capacity', 'life_expectancy', 'cost_of_moving', 'average_maintenance_fee', 'cost_of_energy', 'slots_capacity', 'selling_price', 'lifespan', 'cost']
    for ncol in numerical:
        FLEET[ncol] = FLEET[ncol].str.extract('(\d+\.?\d*)').astype(float)
    
    # LOAD PROBLEM DATA
    servers = pd.read_csv('data/servers.csv')
    selling_prices = pd.read_csv('data/selling_prices.csv')
    demand = pd.read_csv('data/demand.csv')
    datacenters = pd.read_csv('data/datacenters.csv')
    elasticity = pd.read_csv('data/price_elasticity_of_demand.csv')

    base_prices = selling_prices.copy()

    # Define latency to datacenter mapping
    DATA_CENTER_LATENCY = {"low": ["DC1"], "medium": ["DC2"], "high": ["DC3", "DC4"]}

    from scipy.ndimage import gaussian_filter1d

    SIGMA = 20
    d['high'] = np.minimum(gaussian_filter1d(d['high'], sigma=SIGMA), d['high'])
    d['low'] = np.minimum(gaussian_filter1d(d['low'], sigma=SIGMA), d['low'])
    d['medium'] = np.minimum(gaussian_filter1d(d['medium'], sigma=SIGMA), d['medium'])

    # Precompute slotsize_capacity_map
    slotsize_capacity_map = FLEET.groupby(["datacenter_id", "server_generation"])[["capacity", "slots_size"]].sum().to_dict(orient="index")

    for t in range(1, 3):
        print(t)
        D = d[d['time_step'] == t]

        ts_prices = get_time_step_prices(PRICING_STRAT, t)
        selling_prices = update_selling_prices(selling_prices, ts_prices)
        D = update_demand_according_to_prices(D, selling_prices, base_prices, elasticity)

        ts_fleet = []

        for server_gen, demand_row in D.groupby('server_generation'):
            for latency in ['low', 'medium', 'high']:
                dcs = DATA_CENTER_LATENCY[latency]
                curr_demand_cap = demand_row[latency].sum()

                useful_servers = FLEET[(FLEET["server_generation"] == server_gen) & FLEET["datacenter_id"].isin(dcs)]
                # Modify this line to include server_generation in the lookup
                capacity_available = sum(slotsize_capacity_map.get((dc, server_gen), {'capacity': 0})['capacity'] for dc in dcs)
                capacity_needed = max(0, curr_demand_cap - capacity_available) 

                datacenter_max_slots = datacenters[datacenters["latency_sensitivity"] == latency]["slots_capacity"].sum()
                slot_size_remaining = datacenter_max_slots - useful_servers["slots_size"].sum()

                while capacity_needed > 0:
                    for k in dcs:
                        data_center_to_add = k
                        datacenter_curr_slots = FLEET[FLEET["datacenter_id"] == k]["slots_size"].sum()

                        if datacenter_max_slots - datacenter_curr_slots < slot_size_remaining:
                            break

                        filtered_FLEET = FLEET[FLEET["datacenter_id"] == k]
                        FLEET_sorted = filtered_FLEET.sort_values(by='lifespan', ascending=True)

                        FLEET_sorted_row = 0
                        while datacenter_max_slots - datacenter_curr_slots < slot_size_remaining and dcs.index(k) == len(dcs) - 1:
                            ts_fleet.append({
                                "time_step": t,
                                "datacenter_id": dcs[0],
                                "server_generation": FLEET_sorted["server_generation"].iloc[FLEET_sorted_row],
                                "server_id": FLEET_sorted["server_id"].iloc[FLEET_sorted_row],
                                "action": "dismiss"
                            })
                            datacenter_curr_slots -= FLEET_sorted["slots_size"].iloc[FLEET_sorted_row]
                            FLEET_sorted_row += 1
                            data_center_to_add = dcs[0]

                    ts_fleet.append({
                        'time_step': t,
                        'datacenter_id': dcs[0],
                        "server_id": str(uuid1()),
                        "server_generation": server_gen,
                        "action": "buy"
                    })

                    capacity_needed -= servers[servers["server_generation"] == server_gen].iloc[0]["capacity"]

        ts_fleet = pd.DataFrame(data=ts_fleet, columns=cols)
        # ts_fleet = fleet_data_preparation(ts_fleet, servers, datacenters, selling_prices)

        FLEET = update_fleet(t, FLEET, ts_fleet)

    return FLEET[['time_step', 'datacenter_id', "server_id", "server_generation", "action"]], PRICING_STRAT

# Load initial fleet
fleet, pricing_strategy = load_solution('./output/1097.json')
pricing_strategy = pricing_data_preparation(pricing_strategy)
initial_fleet = pd.DataFrame(fleet, columns=["time_step", "datacenter_id", "server_generation", "server_id", "action"])
# Define constant pricing strategy
constant_pricing_strategy = pd.DataFrame(pricing_strategy, columns=['time_step', 'latency_sensitivity', 'server_generation', 'price'])

seeds = [123]

demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    np.random.seed(seed)
    actual_demand = get_actual_demand(demand)
    fleet, pricing_strategy = get_my_solution(actual_demand, initial_fleet, constant_pricing_strategy)
    save_solution(fleet, pricing_strategy, f'./output/{seed}.json')