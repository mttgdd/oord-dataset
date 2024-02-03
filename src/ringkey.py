import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

class RingKey(object):
    def __init__(self) -> None:
        pass

    def call_impl(self, x):
        x = Image.fromarray(x)
        x = x.resize((120, 40))
        x = np.array(x)
        x = np.mean(np.array(x), axis=0)
        return x

    def __call__(self, x):
        device = x.device
        R = []
        for i in range(x.shape[0]):
            x_elem = x[i].squeeze(0).detach().cpu().numpy()
            R.append(self.call_impl(x_elem))
        R = np.array(R)
        R = torch.Tensor(R).to(torch.float).to(device)
        return R