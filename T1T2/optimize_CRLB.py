import simulator
import CRLB
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

IMAGE_PATH = "/home/ubuntu/mri-sim-py.epg/T1T2/images"

def solve_for_CRLB(angles_rad, TE, TR, M0, T1, T2, nitr=20, lr=.00001, SAR=60):
    #angles_rad = torch.nn.Parameter(angles_rad)
    optimizer = torch.optim.SGD([angles_rad], lr=lr)
    T = angles_rad.shape[1]
    loss = np.zeros((nitr,))
    A = torch.tensor(SAR * np.pi / 180. * np.sqrt(T))
    for i in range(nitr):
        #angles_rad.requires_grad_()
        z = CRLB.CRLB_T2(angles_rad, TE, TR, M0, T1, T2).mean()
        optimizer.zero_grad()
        z.backward()
        optimizer.step()
        with torch.no_grad():
            S = torch.norm(angles_rad)
            if S > A:
                angles_rad.data = angles_rad.data / S * A
            loss[i] = z.detach().cpu().numpy()
            print('{}: {:03.4e} SAR={:03.3f}'.format(i, loss[i], S.detach().numpy()))
    return angles_rad, loss

nitr = 20
lr = .00001
SAR = 60 # SAR constraint equiavalent to this constant flip angle

TR = np.array([1000])
T = 32
TE = 10

batch_size = 1
T1 = np.array([1000.]*batch_size).astype(np.float32)
T1_torch = torch.tensor(T1)

T2 = np.linspace(20, 400, batch_size).squeeze().astype(np.float32)
T2 = [100.]
T2_torch = torch.tensor(T2)

M0 = np.array([1.]*batch_size).astype(np.float32)
M0_torch = torch.tensor(M0)

angles_rad = np.pi / 180 * 60 *np.ones((T,))
angles_rad_torch = torch.tensor(angles_rad)[None,:].requires_grad_()
angles_rad_torch_final, loss = solve_for_CRLB(angles_rad_torch, TE, TR, M0_torch, T1_torch, T2_torch, nitr=nitr, lr=lr, SAR=SAR)

FONT_SIZE = 18 

fig = plt.figure();
plt.style.use('seaborn')
ax = plt.axes()
plt.plot(angles_rad.squeeze()*180/np.pi, label='Constant angles')
plt.plot(angles_rad_torch_final.detach().cpu().numpy().squeeze()*180/np.pi, label='Optimized angles')
plt.ylim([0, 150])
plt.xlim(0, 31)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel('Echo Number', fontsize=FONT_SIZE)
plt.ylabel('Angle in degrees', fontsize=FONT_SIZE)
plt.title('Optimized flip angles using Cramer-Rao Lower Bound', fontsize=FONT_SIZE)
plt.legend(frameon=True, fontsize=FONT_SIZE)
plt.savefig(f"{IMAGE_PATH}/crlb_fig1.pdf", bbox_inches="tight")
plt.close('all')

plt.figure();
plt.style.use('seaborn')
plt.plot(simulator.FSE_signal_TR(torch.tensor(angles_rad)[None,:], TE, TR, T1_torch, T2_torch).detach().numpy().squeeze().T, label='Constant angles')
plt.plot(simulator.FSE_signal_TR(angles_rad_torch_final, TE, TR, T1_torch, T2_torch).detach().numpy().squeeze().T, label='Optimized angles')
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel('Echo Number', fontsize=FONT_SIZE)
plt.ylabel('Signal', fontsize=FONT_SIZE)
plt.title('Simulation of Signals', fontsize=FONT_SIZE)
plt.legend(frameon=True, fontsize=FONT_SIZE)
plt.savefig(f"{IMAGE_PATH}/crlb_fig2.pdf", bbox_inches="tight")
plt.close('all')

ax = plt.figure().gca()
plt.style.use('seaborn')
plt.plot(loss)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel('Epochs', fontsize=FONT_SIZE)
plt.ylabel('Training Loss', fontsize=FONT_SIZE)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Training Loss Curve', fontsize=FONT_SIZE)
plt.savefig(f"{IMAGE_PATH}/crlb_fig3.pdf", bbox_inches="tight")
plt.close('all')

T1 = torch.tensor([1000.], dtype=torch.float32)
T2_vals = np.linspace(20, 400, 50)
CRLB_vals_init = np.zeros(T2_vals.shape[0])
CRLB_vals_final = np.zeros(T2_vals.shape[0])
angles_rad_torch = torch.tensor(angles_rad)[None,:]
for i, t2 in enumerate(tqdm(T2_vals)):
    T2 = torch.tensor([t2], dtype=torch.float32)
    CRLB_vals_init[i] = CRLB.CRLB_T2(angles_rad_torch, TE, TR, M0_torch, T1, T2)
    CRLB_vals_final[i] = CRLB.CRLB_T2(angles_rad_torch_final, TE, TR, M0_torch, T1, T2)

plt.figure();
plt.style.use('seaborn')
plt.plot(T2_vals, CRLB_vals_init, label="Constant angles")
plt.plot(T2_vals, CRLB_vals_final, label="Optimized angles")
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel('$T_2$ value (msec)', fontsize=FONT_SIZE)
plt.ylabel('CRLB', fontsize=FONT_SIZE)
plt.title('CRLB($T_2$) vs $T_2$', fontsize=FONT_SIZE)
plt.legend(frameon=True, fontsize=FONT_SIZE)
plt.savefig(f"{IMAGE_PATH}/crlb_fig4.pdf", bbox_inches="tight")
plt.close('all')