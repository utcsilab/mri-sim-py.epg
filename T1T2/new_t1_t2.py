import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

from simulator import FSE_signal_TR

T = 32
TE = 9
TRs = np.array([860, 1830, 2800])
T1_init = 1500.0
T2_init = 200.0
device = torch.device("cuda")
dtype = torch.float32

vals1 = np.linspace(1500, 2500, 100)
vals2 = np.linspace(50, 500, 100)

vals1 = np.repeat(vals1, vals2.shape[0])
vals2 = np.array(vals2.tolist()*100)

T1_vals = torch.tensor(vals1, device=device, dtype=dtype, requires_grad=False)
T2_vals = torch.tensor(vals2, device=device, dtype=dtype, requires_grad=False)
dictionary = []


batch_size = 28000
max_batch = 2000
num_epochs = 500
theta_hat_init_angle = 105.0
step_size_1_init_val = 3 * 1e5
step_size_2_init_val = 3 * 1e6
res_arr = np.ones((288 * 288, 3*T))
t1_arr = np.ones(288 * 288)
t2_arr = np.ones(288 * 288)
pd_arr = np.ones(288 * 288)
pixel_norm = np.ones(288 * 288)
IMAGE_PATH = "images"
FILE_PATH = 'files'

_ridx = np.random.permutation(288 * 288)


class MRIDataset(Dataset):
    """ Decoder MRI dataset with all 180 degrees"""

    # Initialize your data, download, etc.
    def __init__(self):
        rng = np.random.default_rng()
        brain = np.rot90(
            np.abs(np.load("/home/ubuntu/mri-sim-py.epg/T1T2/pics_out_96ETL_experimental_data.npy"))
        ).reshape(288 * 288, 3*T)
        print(brain.shape)
        brain_ridx = brain[_ridx, :]

        self.len = brain_ridx.shape[0]
        self.y_data = brain_ridx

    def __getitem__(self, index):
        return self.y_data[index, :]

    def __len__(self):
        return self.len


def pbnet(y_meas, theta_hat, step_size, TE, TR, loss_arr, dictionary, testFlag=True):
    """
    y_meas: [batch_size, T] -- Input Signal
    theta: [1, T] -- Flip angles
    """
    arr1 = []
    arr2 = []
    with torch.no_grad():
        for i in range(batch_size // max_batch):
            curr_y = y_meas[i*max_batch:i*max_batch+max_batch].detach().cpu()
            dictionary = dictionary.detach().cpu()
            distances = torch.sum((dictionary.T - curr_y[..., None]) ** 2, dim=1)
            min_index = torch.argmin(distances, dim=1)
            del distances
            arr1.append(T1_vals[min_index])
            arr2.append(T2_vals[min_index])

        myt1 = torch.cat(arr1, dim=0)
        myt2 = torch.cat(arr2, dim=0)

    myt1 = myt1.requires_grad_()
    myt2 = myt2.requires_grad_()

    if testFlag:
        y_meas = y_meas.detach()
    sig_est = None
    loss = None
    for kk in tqdm(range(num_epochs)):
        sig_est = FSE_signal_TR(
            theta_hat, TE, TRs, myt1, myt2, B1=1.0).squeeze()
        rho_est = torch.sum(y_meas * sig_est, axis=1) / torch.sum(
            sig_est * sig_est, axis=1
        )
        sig_est = rho_est[:, None] * sig_est
        residual = y_meas - sig_est
        loss = torch.sum(residual ** 2)
        loss_arr.append(float(loss.cpu().detach().numpy()))

        g = torch.autograd.grad(loss, [myt1, myt2], create_graph=not testFlag)
        myt1 = myt1 - step_size[0] * g[0]
        myt2 = myt2 - step_size[1] * g[1]
    return myt1, myt2, sig_est, loss, rho_est, loss_arr


dataset = MRIDataset()
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=2,
    drop_last=False,
    shuffle=False,
)

final_theta = np.ones((1, T)) * theta_hat_init_angle
theta_hat_init = torch.tensor(
    final_theta / 180 * np.pi, dtype=torch.float32).to(device)
theta_hat = theta_hat_init.detach().clone()
theta_hat.requires_grad = True

step_size_init = torch.tensor(
    [step_size_1_init_val, step_size_2_init_val], dtype=torch.float32).to(device)
step_size = step_size_init.detach().clone()
step_size.requires_grad = True

loss_arr = []

with torch.no_grad():
    dictionary = FSE_signal_TR(theta_hat, TE, TRs, T1_vals, T2_vals)

for i, y in tqdm(enumerate(data_loader)):
    y_m = y.to(device)
    y_norm = torch.norm(y_m)
    y_meas = y_m / y_norm

    myt1, myt2, y_est, loss, proton_density, loss_arr = pbnet(
        y_meas, theta_hat, step_size, TE, TRs, loss_arr, dictionary, testFlag=True
    )

    res_arr[i * batch_size: i * batch_size +
            y.shape[0]] = y_est.detach().cpu().numpy()

    t1_arr[i * batch_size: i * batch_size +
           y.shape[0]] = myt1.detach().cpu().numpy()
    t2_arr[i * batch_size: i * batch_size +
           y.shape[0]] = myt2.detach().cpu().numpy()

    pixel_norm[i * batch_size: i * batch_size + y.shape[0]] = (
        y_norm.detach().cpu().numpy()
    )
    pd_arr[i * batch_size: i * batch_size +
           y.shape[0]] = proton_density.detach().cpu().numpy()

print(myt1, myt2)

t1_map_sorted = np.zeros(t1_arr.shape)
t2_map_sorted = np.zeros(t2_arr.shape)
pd_map_sorted = np.zeros(pd_arr.shape)
for i in range(t2_map_sorted.shape[0]):
    t1_map_sorted[_ridx[i]] = t1_arr[i]
    t2_map_sorted[_ridx[i]] = t2_arr[i]
    pd_map_sorted[_ridx[i]] = pd_arr[i]


plt.figure()
plt.imshow(t1_arr.reshape(288, 288), cmap="gray")
plt.colorbar()
plt.savefig(f"{IMAGE_PATH}/T1 Array.png", bbox_inches="tight")

plt.figure()
plt.imshow(t2_arr.reshape(288, 288), cmap="gray")
plt.colorbar()
plt.savefig(f"{IMAGE_PATH}/T2 Array.png", bbox_inches="tight")

brain = np.rot90(np.abs(np.load("pics_out_96ETL_experimental_data.npy"))).reshape(
    288 * 288, 96
)
m1 = np.linalg.norm(brain, axis=1)
mask = m1 > np.max(m1) * 0.05

plt.figure()
plt.imshow(mask.reshape((288, 288)))
plt.savefig(f"{IMAGE_PATH}/mask.png", bbox_inches="tight")

plt.figure()
plt.imshow(t1_map_sorted.reshape(288, 288) * mask.reshape(288, 288), vmax=2500)
plt.axis("off")
plt.colorbar()
plt.savefig(f"{IMAGE_PATH}/t1.png", bbox_inches="tight")

plt.figure()
plt.imshow(t2_map_sorted.reshape(288, 288) * mask.reshape(288, 288))
plt.axis("off")
plt.colorbar()
plt.savefig(f"{IMAGE_PATH}/t2.png", bbox_inches="tight")

plt.figure()
plt.imshow(pd_map_sorted.reshape(288, 288) *
           mask.reshape(288, 288), cmap="gray")
plt.axis("off")
plt.savefig(f"{IMAGE_PATH}/pd.png", bbox_inches="tight")

df = pd.DataFrame()

df['T1'] = t1_map_sorted.ravel()
plt.figure()
sns.histplot(data=df['T1'])
plt.savefig(f"{IMAGE_PATH}/hist_t1.png", bbox_inches="tight")

df['T2'] = t2_map_sorted.ravel()
plt.figure()
sns.histplot(data=df['T2'])
plt.savefig(f"{IMAGE_PATH}/hist_t2.png", bbox_inches="tight")

plt.figure()
plt.plot(loss_arr)
plt.savefig(f"{IMAGE_PATH}/loss.png", bbox_inches="tight")

np.save(f'{FILE_PATH}/mask.npy', mask)
np.save(f'{FILE_PATH}/t1_map_sorted.npy', t1_map_sorted)
np.save(f'{FILE_PATH}/t2_map_sorted.npy', t2_map_sorted)
np.save(f'{FILE_PATH}/pd_map_sorted.npy', pd_map_sorted)
