import torch
import numpy as np

def reproductibility(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm