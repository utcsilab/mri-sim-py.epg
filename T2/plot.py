import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from speed import batches1, batches2 
import torch 
from simulator import FSE_signal_TR

plt.style.use('seaborn')

df = pd.read_csv("Results_full.csv", index_col=0, header=0)
print(df.head())
fig = plt.figure()
ax = plt.axes()
FONT_SIZE = 18

# y = df["Simulator"=="New" & "Device" == "cpu", 'Time'].to_numpy()
# y = df[df['Simulator'].str.contains('New', na = False)]
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
i = 0
sims = {"Parallelized": "New", "Naive": "Old"}
devs = {'cpu': "CPU", 'cuda': "GPU"}
for simulator in ["Parallelized", "Naive"]:
    for device in ["cpu", "cuda"]:
        y = df[df['Simulator'].str.match(sims[simulator]) & df['Device'].str.match(device)]
        x = y.loc[:, 'Batch Size']
        y = y.loc[:, 'Time']
        plt.semilogy(x, y/1e6, label=f'{simulator} EPG on {devs[device]}')
        i += 1

plt.legend(loc=9, bbox_to_anchor=(1.35, 0.75), frameon=True, fontsize=FONT_SIZE)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Time in seconds', fontsize=FONT_SIZE)
plt.title('Batch size vs. time in seconds', fontsize=FONT_SIZE)
plt.savefig('Fig 2 - Time.png', bbox_inches='tight')

plt.close('all')

fig = plt.figure()
ax = plt.axes()
y = np.loadtxt('data/fisher_angles.txt')
ax.hlines(y=60, xmin=0, xmax=y.shape[0], linestyles='dashed')
plt.ylim(0, 180)
plt.xlim(0, 31)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel('Echo Number', fontsize=FONT_SIZE)
plt.ylabel('Angle in degrees', fontsize=FONT_SIZE)
plt.title('Optimized flip angles using Cramer-Rao Lower Bound', fontsize=FONT_SIZE)
plt.plot(list(range(y.shape[0])), y)
plt.savefig('Fig 4b - CRLB.pdf', bbox_inches='tight')
plt.close('all')

T = 32 
TE = 9 
TR = 2800
T1_vals = np.linspace(500, 1000, 1000)   # 100 elements
T2_vals = np.linspace(20, 500, 1000)     # 100 elements

device = torch.device('cpu')

assert T1_vals.shape[0] == T2_vals.shape[0]

angles_rad = torch.ones([1, T], device=device)*60./180.*np.pi # Angles must be in radians
T1 = torch.tensor(T1_vals, dtype=torch.float32, device=device)
T2 = torch.tensor(T2_vals, dtype=torch.float32, device=device)

sig = FSE_signal_TR(angles_rad, TE, TR, T1, T2, B1=1.).squeeze().numpy()
print(T1[:3], T2[:3])

fig = plt.figure()
ax = plt.axes()
plt.xlabel('Echo Number', fontsize=FONT_SIZE)
plt.ylabel('Signal', fontsize=FONT_SIZE)
plt.xlim(0, 31)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
for i in range(100):
    plt.plot(list(range(sig.shape[1])), sig[i*5])
plt.savefig('Fig 1b - signals.pdf', bbox_inches='tight')
plt.close('all')