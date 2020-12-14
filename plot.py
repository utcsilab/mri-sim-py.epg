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
for simulator in ["New", "Old"]:
    for device in ["cpu", "cuda"]:
        y = df[df['Simulator'].str.match(simulator) & df['Device'].str.match(device)]
        x = y.loc[:, 'Batch Size']
        y = y.loc[:, 'Time']
        plt.plot(x, np.log10(y), label=f'{simulator} simulator on {device}', linestyle=linestyles[i])
        i += 1

plt.legend(loc=9, bbox_to_anchor=(1.17, 0.65))
plt.xlabel('Batch Size')
plt.ylabel('$\log_{10}$ Time')
plt.title('Batch size vs. $\log_{10}$ time')
plt.savefig('Time.png', bbox_inches='tight')

plt.close('all')

fig = plt.figure()
ax = plt.axes()
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
i = 0
for simulator in ["New", "Old"]:
    for device in ["cpu", "cuda"]:
        y = df[df['Simulator'].str.match(simulator) & df['Device'].str.match(device)]
        x = y.loc[:, 'Batch Size']
        y = y.loc[:, 'Speed']
        plt.plot(x, y, label=f'{simulator} simulator on {device}', linestyle=linestyles[i])
        i += 1
plt.legend(loc=9, bbox_to_anchor=(1.17, 0.65))
plt.xlabel('Batch Size')
plt.ylabel('Speed in microseconds')
plt.title('Batch size vs. Speed in microseconds')
plt.savefig('Speed.png', bbox_inches='tight')