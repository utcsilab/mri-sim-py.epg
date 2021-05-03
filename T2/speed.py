import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baseline_sim import FSE_signal_TR as sim_old
from simulator import FSE_signal_TR as sim_new

batches1 = [1, 10, 100, 1000, 2000, 5000, 8000, 10000,
           13000, 15000, 18000, 20000, 22000, 25000, 27000, 30000]
batches2 = [1, 10, 100]
devices = ['cpu', 'cuda']

if __name__ == "__main__":
    T = 32
    TE = 9
    TR = 2800

    n = 0

    df1 = pd.DataFrame(columns=["Simulator", "Device", "Batch Size",
                            "Time", "Speed"], index=list(range(len(batches1)*len(devices))))
    for device in devices:
        angles_rad = torch.ones([1, T], device=torch.device(
            device))*170./180.*np.pi  # Angles must be in radians
        for batch in batches1:
            T1_vals = np.linspace(500, 1000, batch)
            T2_vals = np.linspace(20, 500, batch)
            T1 = torch.tensor(T1_vals, dtype=torch.float32,
                            device=torch.device(device))
            T2 = torch.tensor(T2_vals, dtype=torch.float32,
                            device=torch.device(device))

            a = datetime.datetime.now()
            sig = sim_new(angles_rad, TE, TR, T1, T2, B1=1.)
            b = datetime.datetime.now()
            res = (b - a).microseconds
            df1.iloc[n, 0] = "New"
            df1.iloc[n, 1] = device
            df1.iloc[n, 2] = batch
            df1.iloc[n, 3] = res
            df1.iloc[n, 4] = batch / res
            n += 1


    class MRIDataset(Dataset):
        """ MRI dataset."""

        # Initialize your data, download, etc.
        def __init__(self, num):
            num_samples = num
            max_t_val = 501

            rng = np.random.default_rng()
            train = pd.DataFrame(columns=['T1', 'T2'])

            # Considering T1 > T2
            train['T2'] = rng.integers(1, max_t_val, size=num_samples)
            train['T1'] = rng.integers(train['T2'].values, max_t_val*3)

            self.len = train.shape[0]
            self.x_data = train.values

        def __getitem__(self, index):
            return self.x_data[index]

        def __len__(self):
            return self.len


    n = 0
    df2 = pd.DataFrame(columns=["Simulator", "Device", "Batch Size",
                            "Time", "Speed"], index=list(range(len(batches2)*len(devices))))

    for device in devices:
        for batch in batches2:
            dataset = MRIDataset(batch)
            train_loader = DataLoader(dataset=dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=2)
            angles_rad = torch.ones([T], device=torch.device(
                device))*170./180.*np.pi  # Angles must be in radians
            total_time = 0 
            for i, data in enumerate(tqdm(train_loader), 0):
                _data = data.float().reshape(2)
                x = _data.reshape(2)
                t1, t2 = x
                a = datetime.datetime.now()
                sig = sim_old(angles_rad, TE, TR, T1=t1, T2=t2, B1=1.)
                b = datetime.datetime.now()
                total_time += (b - a).microseconds
            df2.iloc[n, 0] = "Old"
            df2.iloc[n, 1] = device
            df2.iloc[n, 2] = batch
            df2.iloc[n, 3] = total_time
            df2.iloc[n, 4] = batch / total_time
            n += 1

    print(df1)
    print(df2)
    df = pd.concat([df1, df2])
    df.to_csv("Results.csv")