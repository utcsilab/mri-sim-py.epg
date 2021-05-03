import numpy as np 
import torch 
from simulator import FSE_signal_TR

# Initializing hyperparameters
T = 32 
TE = 9 
TRs = np.array([860, 1830, 2800])
T1_vals = np.linspace(500, 1000, 100)   # 100 elements
T2_vals = np.linspace(20, 500, 100)     # 100 elements
device = torch.device('cuda' 
                      if torch.cuda.is_available() else 'cpu')

# Ensure both T1_vals and T2_vals have the same 
# number of elements
assert T1_vals.shape[0] == T2_vals.shape[0]

# Convert T1_vals and T2_vals to their corresponding tensors
T1 = torch.tensor(T1_vals, dtype=torch.float32, device=device)
T2 = torch.tensor(T2_vals, dtype=torch.float32, device=device)

# Convert angles from degrees to radians
angles_rad = torch.ones([1, T], device=device)*170./180.*np.pi

# Run EPG simulator
sig = FSE_signal_TR(angles_rad, TE, TRs, T1, T2, B1=1.)
# Returns a tensor with size = torch.Size([100, 32, 1])