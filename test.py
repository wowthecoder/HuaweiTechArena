import pandas as pd

servers = pd.read_csv('./data/servers.csv')
fleet = pd.read_csv('./rl_data/fleet.csv')
demands = pd.read_csv('./rl_data/actual_demand_1061.csv')

demand_ts = demands.loc[demands["time_step"] == 2] 
demand_list = demand_ts.drop(columns="time_step", inplace=False).to_dict(orient='list')
print(demand_list)
print(type(demand_list))

test_id = '71f734a2-32b2-46df-87f7-adceed8ae310'
sgen = fleet.loc[fleet["server_id"] == test_id, 'server_generation'].values[0]

print(sgen)

# Count the frequency of each unique time_step
time_step_counts = demands['time_step'].value_counts()

# Find the maximum frequency
max_frequency_value = time_step_counts.idxmax()
max_frequency = time_step_counts.max()

print(f"The time_step value with the maximum frequency is: {max_frequency_value} (Frequency: {max_frequency})")

