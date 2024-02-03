from config.config import get_cfg_impl
from src.resnet18_netvlad import Net
from src.test import test
from src.compute_distance_matrix import get_embeddings

from sklearn.cluster import KMeans
import torch
import numpy as np
import os

cfg = get_cfg_impl('config/resnet18_netvlad.yaml')
net = Net(cfg).to(cfg['device'])
net.eval()

def calib(net, ref_dataset):
    ref_embeddings = get_embeddings(
        cfg, ref_dataset, 
        lambda x : torch.nn.functional.normalize(net.base_model(x).flatten(start_dim=2).permute(0, 2, 1), p=2, dim=1), 
        None)
    ref_embeddings = np.array(ref_embeddings)
    ref_embeddings = ref_embeddings.reshape(-1, net.dim)
    kmeans_dict = KMeans(
        n_clusters=net.net_vlad.num_clusters,
        init='k-means++', tol=0.0001, n_init=1,
        verbose=1).fit(ref_embeddings)
    net.net_vlad.fit(kmeans_dict.cluster_centers_, ref_embeddings)
    net = net.to(cfg['device'])
    return net

test(cfg, net, None, 'resnet18_netvlad', other_cb=None, calib_cb=calib)