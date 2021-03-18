import torch 
import numpy as np 

a = torch.tensor(np.arange(24).reshape(2, 4, 3))
b = torch.flatten(a, start_dim=1)[:, :, None]
print(a.shape, b.shape)