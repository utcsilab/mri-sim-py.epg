import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from speed import batches1, batches2 

plt.style.use('seaborn')

df = pd.read_csv("Results_full.csv", index_col=0, header=0)
print(df.head())
fig = plt.figure()
ax = plt.axes()

# y = df["Simulator"=="New" & "Device" == "cpu", 'Time'].to_numpy()
# y = df[df['Simulator'].str.contains('New', na = False)]
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
i = 0
sims = {"Parallelized": "New", "Naive": "Old"}
for simulator in ["Parallelized", "Naive"]:
    for device in ["cpu", "cuda"]:
        y = df[df['Simulator'].str.match(sims[simulator]) & df['Device'].str.match(device)]
        x = y.loc[:, 'Batch Size']
        y = y.loc[:, 'Time']
        plt.semilogy(x, y, label=f'{simulator} EPG on {device.upper()}')
        i += 1

plt.legend(loc=9, bbox_to_anchor=(1.2, 0.65), frameon=True)
plt.xlabel('Batch Size')
plt.ylabel('Time in microseconds')
plt.title('Batch size vs. time in microseconds')
plt.savefig('Time.png', bbox_inches='tight')

plt.close('all')

fig = plt.figure()
ax = plt.axes()
linestyles = ['solid', 'dashdot', 'dashed', 'dotted']
i = 0
sims = {"Parallelized": "New", "Naive": "Old"}
for simulator in ["Parallelized", "Naive"]:
    for device in ["cpu", "cuda"]:
        y = df[df['Simulator'].str.match(sims[simulator]) & df['Device'].str.match(device)]
        x = y.loc[:, 'Batch Size']
        y = y.loc[:, 'Speed']
        plt.plot(x, y, label=f'{simulator} EPG on {device.upper()}', linestyle=linestyles[i])
        i += 1
plt.legend(loc=9, bbox_to_anchor=(1.2, 0.65), frameon=True)
plt.xlabel('Batch Size')
plt.ylabel('Samples processed per microsecond')
plt.title('Batch size vs. Samples processed per microsecond')
plt.savefig('Speed.png', bbox_inches='tight')