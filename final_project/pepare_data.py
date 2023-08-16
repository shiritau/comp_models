import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from glob import glob
import random
from tqdm import tqdm
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
random.seed(42)
np.seed = 42

def save_dcm_as_npys(dir):
    for patient in os.listdir(dir):
        patient_path = os.path.join(dir, patient)
        t1_path = os.path.join(patient_path, 'T1w')
        scan_array = []
        for file in os.listdir(t1_path):

            dicom_path = os.path.join(t1_path, file)

            # Load the DICOM file
            dicom_data = pydicom.dcmread(dicom_path)

            # Convert the DICOM pixel data to a NumPy array
            pixel_array = dicom_data.pixel_array

            # Append the pixel array to the list
            scan_array.append(pixel_array)

        # Convert the list of pixel arrays into a NumPy array
        image_stack = np.array(scan_array)


def train_test_split():
    pass

def show_pos_seg(data_dir):
    for scan in os.listdir(data_dir):
        if 'data' in scan:
            scan_path = os.path.join(data_dir, scan)
            seg_path = scan_path.replace('data', 'seg')
            scan = np.load(scan_path)
            seg = np.load(seg_path)
            plt.subplot(1, 2, 1)
            plt.imshow(scan, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(seg)
            plt.show()

def create_csv_of_paths(data_dir, npy_dir):
    df = pd.DataFrame(columns=['path', 'label', 'shape'])
    t1_paths = glob(os.path.join(data_dir, '*/*t1.nii.gz'))
    for scan_path in tqdm(t1_paths[:100]):
        name = scan_path.split('\\')[-2]
        scan_data = nib.load(scan_path).get_fdata()
        seg_path = scan_path.replace('t1', 'seg')
        seg_data = nib.load(seg_path).get_fdata()
        for slice_num in range(scan_data.shape[2]):
            slice_scan = scan_data[:, :, slice_num]
            slice_seg = seg_data[:, :, slice_num]
            if np.sum(slice_seg) == 0:
                label = 0
                np.save(os.path.join(npy_dir, str(label), f'{name}_data_{slice_num}.npy'), slice_scan)
                np.save(os.path.join(npy_dir, str(label), f'{name}_seg_{slice_num}.npy'), slice_seg)
            else:
                label = 1
                np.save(os.path.join(npy_dir,str(label), f'{name}_data_{slice_num}.npy'), slice_scan)
                np.save(os.path.join(npy_dir,str(label), f'{name}_seg_{slice_num}.npy'), slice_seg)
            scan_shape = slice_scan.shape
            df = df.append({'path': os.path.join(npy_dir, str(label), f'{name}_data_{slice_num}.npy'), 'label': label, 'shape': scan_shape}, ignore_index=True)

    return df


def split_train_val(df):
    # 80% train, 20% val
    df['splt'] = df.apply(lambda x: 'train' if np.random.rand() < 0.8 else 'val', axis=1)
    return df


if __name__ == '__main__':
    ROOT_DIR = r'C:\Users\shiri\Documents\School\Master\Courses\Comp_models_of_learning\final_project\brats'
    DATA_DIR = f'{ROOT_DIR}/training'
    NPY_DIR = f'{ROOT_DIR}/npys'
    df = create_csv_of_paths(DATA_DIR, NPY_DIR)
    df = split_train_val(df)
    df.to_csv(f'{ROOT_DIR}/paths.csv', index=False)
    #show_pos_seg(f'{NPY_DIR}/1')