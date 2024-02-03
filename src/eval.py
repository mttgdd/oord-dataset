import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

def do_eval(cfg, loc_dataset, ref_dataset, distances=None, DB=50):
    

    loc_pos = np.array(list(loc_dataset.pos.values()))
    ref_pos = np.array(list(ref_dataset.pos.values()))

    loc_ref_se2_dist = cdist(loc_pos, ref_pos)
    positives = loc_ref_se2_dist < cfg['gps_match_tolerance']

    rcs = {}
    asst = np.argsort(distances)
    for k in tqdm(range(1, DB)):
        rc = 0
        psds = []
        for i in range(distances.shape[0]):
            sd = asst[i, :k]
            psd = [positives[i, s] for s in sd]
            rc += int(np.any(psd))
            psds.append(int(np.sum(psd)))
        rc /= distances.shape[0]
        rcs[k] = rc

    return rcs