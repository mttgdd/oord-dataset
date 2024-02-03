from config.config import get_cfg_impl
from src.test import test
from src.models import get_nn

import torch

cfg = get_cfg_impl('config/alexnet.yaml')

net = get_nn('alexnet')
net = net.to(cfg['device'])
net.eval()

test(cfg, net, None, 'alexnet')
	
