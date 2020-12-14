import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from speed import batches1, batches2 

plt.style.use('seaborn-whitegrid')

df = pd.read_csv("Results.csv", index_col=0, header=0)
print(df.head())
fig = plt.figure()
ax = plt.axes()

# y = df["Simulator"=="New" & "Device" == "cpu", 'Time'].to_numpy()
# y = df[df['Simulator'].str.contains('New', na = False)]
for simulator in ["New", "Old"]:
    for device in ["cpu", "cuda"]:
        y = df[df['Simulator'].str.match(simulator) & df['Device'].str.match(device)]
        x = y.loc[:, 'Batch Size']
        y = y.loc[:, 'Time']
        plt.plot(x, y, label=f'{simulator} simulator on {device}')
plt.legend();
plt.xlabel('Batch Size')
plt.ylabel('Time')
plt.savefig('Time.png')

plt.close('all')

fig = plt.figure()
ax = plt.axes()
for simulator in ["New", "Old"]:
    for device in ["cpu", "cuda"]:
        y = df[df['Simulator'].str.match(simulator) & df['Device'].str.match(device)]
        x = y.loc[:, 'Batch Size']
        y = y.loc[:, 'Speed']
        plt.plot(x, y, label=f'{simulator} simulator on {device}')
plt.legend();
plt.xlabel('Batch Size')
plt.ylabel('Speed')
plt.savefig('Speed.png')