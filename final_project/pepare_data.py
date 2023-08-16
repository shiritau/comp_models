import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
import random
from tqdm import tqdm
from scipy.ndimage import zoom

random.seed(42)
np.seed = 42


def show_segmentation(data_dir: str):
    paths = glob(os.path.join(data_dir, '*data*'))
    for scan_path in paths:
        seg_path = scan_path.replace('data', 'seg')
        scan = np.load(scan_path)
        seg = np.load(seg_path)
        plt.subplot(1, 2, 1)
        plt.imshow(scan, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(seg)
        plt.show()


def prepare_data(data_dir: str, npy_dir: str, add_zoom: bool = False):
    df = pd.DataFrame(columns=['path', 'label', 'shape'])
    t1_paths = glob(os.path.join(data_dir, '*/*t1.nii.gz'))

    for scan_path in tqdm(t1_paths[:100]):  # only take first 100 scans
        name = scan_path.split('\\')[-2]
        scan_data = nib.load(scan_path).get_fdata()
        seg_path = scan_path.replace('t1', 'seg')
        seg_data = nib.load(seg_path).get_fdata()
        for slice_num in range(scan_data.shape[2]):
            slice_scan = scan_data[:, :, slice_num]
            if add_zoom:
                slice_scan = zoom(slice_scan, (0.25,0.25))
            slice_seg = seg_data[:, :, slice_num]
            if np.sum(slice_seg) == 0:
                label = 0
                np.save(os.path.join(npy_dir, str(label), f'{name}_data_{slice_num}.npy'), slice_scan)
                np.save(os.path.join(npy_dir, str(label), f'{name}_seg_{slice_num}.npy'), slice_seg)
            else:
                label = 1
                np.save(os.path.join(npy_dir, str(label), f'{name}_data_{slice_num}.npy'), slice_scan)
                np.save(os.path.join(npy_dir, str(label), f'{name}_seg_{slice_num}.npy'), slice_seg)
            scan_shape = slice_scan.shape
            df = df.append({'path': os.path.join(npy_dir, str(label), f'{name}_data_{slice_num}.npy'), 'label': label,
                            'shape': scan_shape}, ignore_index=True)

    return df


def split_train_val(df):
    df['splt'] = df.apply(lambda x: 'train' if np.random.rand() < 0.8 else 'val', axis=1)
    return df


if __name__ == '__main__':
    ROOT_DIR = r'C:\Users\shiri\Documents\School\Master\Courses\Comp_models_of_learning\final_project\brats'
    DATA_DIR = f'{ROOT_DIR}/training'
    NPY_DIR = f'{ROOT_DIR}/npys'
    data_df = prepare_data(DATA_DIR, NPY_DIR, add_zoom=True)
    data_df = split_train_val(data_df)
    data_df.to_csv(f'{ROOT_DIR}/paths.csv', index=False)
    # show_pos_seg(f'{NPY_DIR}/1')
