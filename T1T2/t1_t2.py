# Aim
# ### The objective of this notebook is to estimate the T1, T2 and PD from the experimental data using the variable projection method for the 2D images.

from __future__ import division

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import grad
from tqdm import tqdm

from simulator import FSE_signal, FSE_signal_TR

pics_out1 = np.load("pics_out_96ETL_experimental_data.npy")
IMAGE_PATH = "images"

plt.figure(figsize=(16, 12))
for index in range(3):
    plt.subplot(1, 3, index + 1)
    plt.imshow(
        np.abs(pics_out1[:, :, 0 + 32 * index]) > 0.00002, cmap="gray"
    )  # ,vmax = 800000,vmin=0)
    plt.title("Echo number: " + str(0 + 32 * index))
plt.savefig(f"{IMAGE_PATH}/masks1.png", bbox_inches="tight")
print(index)
mask = (
    np.abs(pics_out1[:, :, 0 + 32 * index]) > 0.00002
) * 1  # mask of brain where the actual images is present
plt.figure(figsize=(16, 12))
plt.imshow(mask, cmap="gray")
plt.savefig(f"{IMAGE_PATH}/masks2.png", bbox_inches="tight")

ETL = pics_out1.shape[2]
pics_out = np.zeros(pics_out1.shape, dtype=complex)
signal_evolution = np.zeros((36196, pics_out1.shape[2]), dtype=complex)

for index in range(ETL):
    Im_synthetic = np.zeros([pics_out1.shape[0], pics_out1.shape[1]], dtype=complex)
    data = pics_out1[:, :, index]
    Im_synthetic[np.nonzero(mask)] = data[np.nonzero(mask)]
    pics_out[:, :, index] = Im_synthetic
    signal_evolution[:, index] = Im_synthetic[np.nonzero(mask)]
print(signal_evolution.shape)

plt.figure(figsize=(16, 12))
for index in range(8):
    plt.subplot(2, 4, index + 1)
    plt.imshow(np.abs(pics_out[:, :, index * 12]), cmap="gray", vmax=0.0001, vmin=0)
    plt.title("Echo number: " + str(12 * index))
plt.savefig(f"{IMAGE_PATH}/progress.png", bbox_inches="tight")

device = torch.device("cuda")


class LeastSquaresRegressorTorch1:
    def __init__(self, n_iter=10, eta=0.1):
        self.n_iter = n_iter  # number of iterations
        self.eta = eta  # the step size

    def fit(self, Y, t1, t2):
        Yt = torch.tensor(
            Y, dtype=torch.float, device=torch.device("cuda")
        )  # convert numpy to torch data files
        self.t2 = torch.tensor(
            t2, dtype=torch.float32, requires_grad=True, device=torch.device("cuda")
        )
        self.t1 = torch.tensor(
            t1, dtype=torch.float32, requires_grad=True, device=torch.device("cuda")
        )

        self.history = []
        optimizer = torch.optim.SGD([self.t1, self.t2], lr=self.eta)
        ETL = 32  # echo train length,
        TE = 9.1  # echo time
        TRs = np.array([860, 1830, 2800])
        self.initial_angle = torch.tensor(
            [180], dtype=torch.float32, requires_grad=True, device=torch.device("cuda")
        )
        M0 = 1
        angles_rad = torch.tensor(
            torch.ones((1, ETL), device=torch.device("cuda"))
            * self.initial_angle
            * math.pi
            / 180,
            device=torch.device("cuda"),
            dtype=torch.float32,
        )
        for i in range(self.n_iter):
            total_loss = 0
            # print(self.t1,self.t2)
            X_sim_CSF = FSE_signal_TR(angles_rad, TE, TRs, T1=self.t1, T2=self.t2)
            loss = (X_sim_CSF / torch.max(X_sim_CSF)).reshape(
                (self.t1.shape[0], 3 * ETL)
            ) - Yt
            loss_batch = torch.sum(loss ** 2)
            optimizer.zero_grad()  # reset all gradients
            loss_batch.backward()  # compute the gradients for the loss for this batch
            optimizer.step()
            total_loss += loss_batch.item()
            self.history.append(total_loss)

        # print('SGD-minibatch final loss: {:.4f}'.format(total_loss))
        return self.t1, self.t2, self.initial_angle


# regr = LeastSquaresRegressorTorch1(n_iter=50, eta=1000)

# data_avg_CSF = np.abs(signal_evolution[20000,0:96])

# t1_CSF, t2_CSF, initial_angle_CSF = regr.fit(data_avg_CSF/np.max(data_avg_CSF), t1 = [500], t2=[200])
# print(t1_CSF,t2_CSF,initial_angle_CSF)

# plt.figure(figsize=(16,12))
# plt.plot(regr.history, '.-')
# plt.title('CSF loss curve')
# plt.savefig(f'{IMAGE_PATH}/CSF_loss.png', bbox_inches='tight')

regr = LeastSquaresRegressorTorch1(n_iter=20, eta=10000)

t1_min = 100
t1_max = 6000
t2_min = 10
t2_max = 1000
N_t1 = 300
N_t2 = 550
initial_angle = 180.0
TE = 9.1  # echo time
TRs = np.array([860.0, 1830.0, 2800.0])

t1_vals = torch.tensor(np.linspace(t1_min, t1_max, N_t1))
t2_vals = torch.tensor(np.linspace(t2_min, t2_max, N_t2))

print("T1 step:", t1_vals[1] - t1_vals[0], "ms")
print("T2 step:", t2_vals[1] - t2_vals[0], "ms")

_t1_vals, _t2_vals = np.meshgrid(t1_vals, t2_vals)
_t1_t2_vals = np.stack((_t1_vals.ravel(), _t2_vals.ravel()), axis=1)
_idxs = _t1_t2_vals[:, 0] >= _t1_t2_vals[:, 1]
t1_t2_vals = torch.tensor(_t1_t2_vals[_idxs, :], dtype=torch.float32)


angles_rad = torch.ones((1, 32), dtype=torch.float32) * initial_angle * math.pi / 180
X_sim_CSF = FSE_signal_TR(angles_rad, TE, TRs, T1=t1_t2_vals[:, 0], T2=t1_t2_vals[:, 1])
X_sim_CSF = X_sim_CSF / torch.norm(X_sim_CSF, dim=1, keepdim=True)

def dictionary_match(y_sig, Dict, t1t2_list):
    # get best match
    # print(y_sig.shape)
    # z = torch.mm(Dict.squeeze(), y_sig.T)
    # i = torch.argmax(torch.abs(z), dim=0)

    distances = torch.sum((Dict - y_sig.T[None, ...]) ** 2, dim=1)

    i = torch.argmin(distances, dim=1)
    return distances[i], t1t2_list[i, :], i  # , sc


t1t2_all = np.zeros(signal_evolution.shape[0])


a, t1t2, c = dictionary_match(
    torch.tensor(np.abs(signal_evolution), dtype=torch.float32),
    X_sim_CSF.to(torch.float32),
    t1_t2_vals,
)
T1_vals = np.ones((signal_evolution.shape[0])) * 500.0
T2_vals = np.ones((signal_evolution.shape[0])) * 500.0
N = T2_vals.shape[0]

# data_avg_CSF = np.abs(signal_evolution[index, 0:96])
# t1_CSF, t2_CSF, initial_angle_CSF = regr.fit(
#     data_avg_CSF / np.max(data_avg_CSF), t1=T1_vals, t2=T2_vals
# )


# Plot T1 and T2 map
Im_synthetic1 = np.zeros([pics_out1.shape[0], pics_out1.shape[1]])
Im_synthetic1[np.nonzero(mask)] = T1_vals

plt.figure(figsize=(16, 12))
plt.imshow(Im_synthetic1, cmap="gray", vmax=8000, vmin=100)
plt.title("T1 map")
plt.savefig(f"{IMAGE_PATH}/T1_map.png", bbox_inches="tight")

Im_synthetic1 = np.zeros([pics_out1.shape[0], pics_out1.shape[1]])
Im_synthetic1[np.nonzero(mask)] = T2_vals

plt.figure(figsize=(16, 12))
plt.imshow(Im_synthetic1, cmap="gray", vmax=1000, vmin=10)
plt.title("T2 map")
plt.savefig(f"{IMAGE_PATH}/T2_map.png", bbox_inches="tight")

print(Im_synthetic1[128:150, 128:150])