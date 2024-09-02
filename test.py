import pandas as pd

df = pd.read_csv('./data/servers.csv')
fleet = pd.read_csv('./rl_data/fleet.csv')

rtime1, rtime2 = df.loc[3, 'release_time'].strip('[]').split(',')
print(type(rtime1))

print("rtime1:", rtime1, "rtime2:", rtime2)

test_id = '71f734a2-32b2-46df-87f7-adceed8ae310'
sgen = fleet.loc[fleet["server_id"] == test_id, 'server_generation'].values[0]

print(sgen)