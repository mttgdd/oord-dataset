from src.eval import do_eval
from src.compute_distance_matrix import get_distances
from src.dataset import Dataset
from config.config import get_data_yaml

import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch

def test(
    cfg, embedding_fn, distance_fn, 
    csv_out, other_cb=None, calib_cb=None, yaml_file='config/test.yaml'):

    os.makedirs(cfg['out_dir'], exist_ok=True)

    ave_rc1 = 0.0
    n = 0
    
    for data in get_data_yaml(yaml_file):

        ref_dataset = Dataset(
            cfg, data['dir'], data['ref_exp'])
        
        # Optionally calibrate against the reference(map) data
        if calib_cb: 
            embedding_fn = calib_cb(embedding_fn, ref_dataset)
            # Save weights
            pth = os.path.join(cfg['out_dir'], f"resnet18_netvlad_oord-{data['ref_exp']['date_str']}.pth")
            print(f'Saving pth: {pth}')
            torch.save(embedding_fn.state_dict(), pth)

        other = other_cb(ref_dataset) if other_cb else None

        for loc_exp in data['loc_exps']:
            # Path to output file
            if csv_out:
                csv_file = os.path.join(
                    cfg['out_dir'], f"{csv_out}_{data['ref_exp']['date_str']}_{loc_exp['date_str']}.csv")

                # Abort if run already
                if os.path.exists(csv_file):
                    print(f'WARNING. {csv_file} exists, skipping this setting!')
                    continue

                print(f'{csv_file} does not exist, running this setting!')

            # Datasets
            # TODO: use TrainingDataset
            loc_dataset = Dataset(
                cfg, data['dir'], loc_exp)

            # Get distances
            if other_cb:
                distances = get_distances(cfg,
                    loc_dataset, ref_dataset,
                    embedding_fn=embedding_fn, distance_fn=distance_fn, other=other)
            else:
                distances = get_distances(cfg,
                    loc_dataset, ref_dataset,
                    embedding_fn=embedding_fn, distance_fn=distance_fn)
                
            # Save distances
            if csv_out:
                png_file = csv_file.replace('.csv', '.png')
                plt.imshow(distances)
                plt.savefig(png_file)
            
            # Do evaluation
            rcs = do_eval(cfg, loc_dataset, ref_dataset, distances)
            # print(rcs[1])

            ave_rc1 += rcs[1]
            n += 1

            # Save results
            if csv_out:
                df = pd.DataFrame.from_dict([rcs])
                df.to_csv(csv_file)

    return ave_rc1/n