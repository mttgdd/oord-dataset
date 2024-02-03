import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import torch
import scipy

def get_embeddings(cfg, dataset, embedding_fn, other):
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=12
    )

    embeddings = []
    for x in tqdm(
        loader, total=len(loader), 
        desc=f'Embedding'):
        x = x.to(cfg['device'])
        z = embedding_fn(x, other) if other else embedding_fn(x)
        embeddings.extend(z.tolist())

    return embeddings

def get_distances(cfg, loc_dataset, ref_dataset, 
    embedding_fn=None, distance_fn=None, other=None):


    loc_embeddings = get_embeddings(cfg, loc_dataset, embedding_fn, other)
    ref_embeddings = get_embeddings(cfg, ref_dataset, embedding_fn, other)

    print(f'Computing distances from {loc_dataset.date_str} to {ref_dataset.date_str}')
    if distance_fn is None:
        loc_embeddings = np.array(loc_embeddings)
        ref_embeddings = np.array(ref_embeddings)
        if len(loc_embeddings.shape) == 1:
            loc_embeddings = loc_embeddings.reshape(loc_embeddings.shape[0],1)
            ref_embeddings = ref_embeddings.reshape(ref_embeddings.shape[0],1)
        distances = scipy.spatial.distance.cdist(
            loc_embeddings, ref_embeddings,
            metric='euclidean'
        )
    else:
        def distance_cb(loc_idx, ref_idx, loc_feats, ref_feats):
            loc_idx, ref_idx = int(loc_idx[0]), int(ref_idx[0])
            loc_feat, ref_feat = loc_feats[loc_idx], ref_feats[ref_idx]
            return distance_fn(loc_feat, ref_feat)

        distances = pairwise_distances(
            np.array(range(len(loc_embeddings))).reshape(-1, 1),
            np.array(range(len(ref_embeddings))).reshape(-1, 1),
            n_jobs=16, metric=lambda loc_idx, ref_idx: distance_cb(
                loc_idx, ref_idx, loc_embeddings, ref_embeddings)
        )

    return distances