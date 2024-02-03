import os
from PIL import Image
import numpy as np
import torch
import pandas as pd
from scipy.spatial import cKDTree

from src.radar import load_radar, radar_polar_to_cartesian

class Dataset:
    def __init__(
            self, cfg, dir, exp,
            open=True):
        
        self.cfg = cfg
        self.date_str = exp['date_str']
        self.dir = dir

        # Some files to be read
        self.date_str_dir = os.path.join(
            self.cfg['data_dir'], dir, self.date_str)
        self.radar_timestamps_file = os.path.join(
            self.cfg['tar_dir'], 
            self.date_str, "Navtech_CTS350-X_Radar.timestamps")
        self.microstrain_file = os.path.join(
            self.cfg['data_dir'], dir,
            self.date_str, "MicrostrainMIP/gps.csv")
        self.radar_dir = os.path.join(
            self.cfg['data_dir'], dir,
            self.date_str,
            "Navtech_CTS350-X_Radar")
        
        start = 0 if not 'start' in exp else exp['start']
        end = -1 if not 'end' in exp else exp['end']
        if open: self.open(start=start, end=end)

    def open(self, start=0, end=-1):        
        # Read radar timestamps
        self.radar_timestamps = Dataset.get_radar_timestamps(
            self.radar_timestamps_file)

        # Downsample strategy
        self.radar_timestamps = self.radar_timestamps[::self.cfg['downsample']]

        # Crop timestamps
        self.radar_timestamps = self.radar_timestamps[start:end]

        # Radar pos
        self.microstrain_df = pd.read_csv(self.microstrain_file)
        self.pos = Dataset.get_radar_positions(
            self.microstrain_df, self.radar_timestamps
        )

    @staticmethod
    def get_radar_timestamps(radar_timestamps_file):
        radar_timestamps = []
        with open(radar_timestamps_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                radar_timestamps.append(
                    int(line.strip('\n').split(' ')[0]))
        return radar_timestamps
    
    def load_image(self, png_file):
        
        # Read fft data, ignore meta data
        timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(png_file)

        # Suppress early/late range returns
        fft_data[:, :self.cfg['min_bin']] = 0
        fft_data = fft_data[:, :self.cfg['max_bin']]

        # Crop ranges optionally
        if self.cfg['cartesian'] == False:
            img = np.squeeze(fft_data, 2)

            # Crop and resize
            img = img[:, :self.cfg['max_bin']]
            img = Image.fromarray(img)

            img = img.resize((self.cfg['num_bins'], self.cfg['num_azis']))

            img = np.array(img)
        else: # Convert to cartesian optionally
            # TODO: is azimuths broken?
            img = radar_polar_to_cartesian(
                azimuths, fft_data, radar_resolution, 
                self.cfg['cart_res'], 
                self.cfg['cart_pw'], True)
            img = img.squeeze(-1)

        # Fft
        if self.cfg['fft']:
            img = np.fft.fft(img)
            img = np.abs(img)
            img = img.astype(np.float32)

        return img
    
    def prepare_output_tensor(self, img):
        # Scaling by 255 already done in load_radar
        img = torch.Tensor(img).to(torch.float)
        img = img.unsqueeze(0)

        if self.cfg['channels'] > 1:
            img = img.repeat(self.cfg['channels'], 1, 1)

        return img

    def get_png_file(self, idx):
        radar_timestamp = self.radar_timestamps[idx]
        png_file = os.path.join(
            self.cfg['data_dir'], self.dir,
            self.date_str,
            "Navtech_CTS350-X_Radar", 
            f"{radar_timestamp}.png")
        return radar_timestamp, png_file

    @staticmethod
    def get_radar_positions(microstrain_df, radar_timestamps):

        # Use kd-tree for fast lookup
        gt_tss = microstrain_df.timestamp.to_numpy()
        keys = np.expand_dims(gt_tss, axis=-1)
        tree = cKDTree(keys)
        query = np.array(radar_timestamps)
        query = np.expand_dims(query, axis=-1)
        _, out = tree.query(query)
        gt_idxs = out.tolist()

        # Build output
        pos = {}
        for radar_timestamp, gt_idx in zip(radar_timestamps, gt_idxs):
            pos[radar_timestamp] = np.array(
                (microstrain_df.iloc[gt_idx].utm_northing, 
                microstrain_df.iloc[gt_idx].utm_easting))
        
        return pos
    
    def __len__(self):
        return len(self.radar_timestamps)

    def __getitem__(self, idx):

        # Image filename
        _, png_file = self.get_png_file(idx)

        # Read and crop out meta data
        img = self.load_image(png_file)

        return self.prepare_output_tensor(img)