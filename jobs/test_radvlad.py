from config.config import get_cfg_impl
from src.test import test
from src.radvlad import do_cluster, get_vlad

from tqdm import tqdm
import numpy as np
import torch

def call(x, kmeans_dict):
     device = x.device
     R = []
     for i in tqdm(range(x.shape[0])):
          x_elem = x[i].squeeze(0).detach().cpu().numpy()
          x_elem = get_vlad(cfg, kmeans_dict, x_elem)
          R.append(x_elem)
     R = np.array(R)
     R = torch.Tensor(R).to(torch.float).to(device)
     return R

cfg = get_cfg_impl('config/radvlad.yaml')

test(cfg, call, None, 'radvlad', other_cb=lambda d: do_cluster(cfg, d))