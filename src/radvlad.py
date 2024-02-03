from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans 

def do_cluster(cfg, dataset):        
    
    descs = []
    for img in tqdm(
        dataset, total=len(dataset), desc=f"Clustering"):
        for i in range(img.shape[0]):
            x = img[i].squeeze(0).detach().cpu().numpy()
            descs.extend(x.tolist())
    descs = np.array(descs).astype(np.float32)

    kmeans_dict = KMeans(
        n_clusters=cfg['num_clusters'],
        init='k-means++', tol=0.0001, n_init=1,
        verbose=1).fit(descs)
    
    return kmeans_dict

def get_vlad(cfg, kmeans_dict, img):
    vlad = np.zeros([cfg['num_clusters'], img.shape[1]])
    cluster_ids = kmeans_dict.predict(img)
    for j in range(img.shape[0]):
        vlad[cluster_ids[j], :] += img[j, :] - kmeans_dict.cluster_centers_[cluster_ids[j], :]
    vlad = vlad.flatten()
    vlad = np.sign(vlad)*np.sqrt(np.abs(vlad))
    vlad = vlad/np.sqrt(np.dot(vlad,vlad))
    return vlad
