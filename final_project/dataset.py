import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MLPDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scan_path = self.df.loc[idx, 'path']
        label = self.df.loc[idx, 'label']
        scan = np.load(scan_path)
        # clip between 0 and 2000
        scan = np.clip(scan, 0, 800)
        # normalize
        scan = (scan - np.min(scan)) / (np.max(scan) - np.min(scan) + 0.00001)

        # For efficientnet, make 3 channels
        # scan = np.stack((scan,)*3, axis=0)

        sample = {'scan': scan, 'label': label}
        return sample


def get_datasets(df_path):
    df = pd.read_csv(df_path)
    train_dataset = MLPDataset(df[df['splt'] == 'train'])
    val_dataset = MLPDataset(df[df['splt'] == 'val'])

    return train_dataset, val_dataset
