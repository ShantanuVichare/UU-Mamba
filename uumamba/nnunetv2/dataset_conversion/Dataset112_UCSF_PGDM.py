import multiprocessing
import os
import time
import shutil
import gzip
from pathlib import Path
import random
import json

import concurrent.futures


import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def copy_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    # 1 (red) is TC, label 2 (green) is Edema (ED), label 4 is Enhancing (ET)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


# def convert_labels_back(seg: np.ndarray):
#     new_seg = np.zeros_like(seg)
#     new_seg[seg == 1] = 2
#     new_seg[seg == 3] = 4
#     new_seg[seg == 2] = 1
#     return new_seg


# def load_convert_labels_back(filename, input_folder, output_folder):
#     a = sitk.ReadImage(join(input_folder, filename))
#     b = sitk.GetArrayFromImage(a)
#     c = convert_labels_back(b)
#     d = sitk.GetImageFromArray(c)
#     d.CopyInformation(a)
#     sitk.WriteImage(d, join(output_folder, filename))


# def convert_folder_with_preds_back_to_orig_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
#     """
#     reads all prediction files (nifti) in the input folder, converts the labels back to orig convention and saves the
#     """
#     maybe_mkdir_p(output_folder)
#     nii = subfiles(input_folder, suffix='.nii.gz', join=False)
#     with multiprocessing.get_context("spawn").Pool(num_processes) as p:
#         p.starmap(load_convert_labels_back, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


def make_out_dirs(out_dir: Path):
    out_train_dir = out_dir / "imagesTr"
    out_trainlabels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"
    out_testlabels_dir = out_dir / "labelsTs"
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_trainlabels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    os.makedirs(out_testlabels_dir, exist_ok=True)
    
    return out_dir, out_train_dir, out_trainlabels_dir, out_test_dir, out_testlabels_dir


def convert_to_nnunet(src_data_folder: Path, target_data_folder: Path, train_test_split = 0.8):

    # Get all cases
    case_ids = [c for c in os.listdir(src_data_folder) if c.startswith('UCSF-PDGM-')]

    # Split 
    random.shuffle(case_ids)
    train_size = int(len(case_ids) * train_test_split)
    train_cases = case_ids[:train_size]
    test_cases = case_ids[train_size:]
    
    out_dir, train_dir, trainlabels_dir, test_dir, testlabels_dir = make_out_dirs(target_data_folder)

    # setting up nnU-Net folders
    def transfer_files(case_ids, images_dir, labels_dir):
        for c in case_ids:
            shutil.copy(join(src_data_folder, c, c + "_t1.nii.gz"), join(images_dir, c + '_0000.nii.gz'))
            shutil.copy(join(src_data_folder, c, c + "_t1ce.nii.gz"), join(images_dir, c + '_0001.nii.gz'))
            shutil.copy(join(src_data_folder, c, c + "_t2.nii.gz"), join(images_dir, c + '_0002.nii.gz'))
            shutil.copy(join(src_data_folder, c, c + "_flair.nii.gz"), join(images_dir, c + '_0003.nii.gz'))

            copy_segmentation_and_convert_labels_to_nnUNet(
                join(src_data_folder, c, c + "_seg.nii.gz"), join(labels_dir, c + '.nii.gz')
            )
        return len(case_ids)

    num_training_cases = transfer_files(train_cases, train_dir, trainlabels_dir)
    num_testing_cases = transfer_files(test_cases, test_dir, testlabels_dir)

    generate_dataset_json(out_dir,
                          channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
                          labels={
                              'background': 0,
                              'whole tumor': (1, 2, 3),
                              'tumor core': (2, 3),
                              'enhancing tumor': (3, )
                          },
                          num_training_cases=num_training_cases,
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3),
                          license='mock license',
                          reference='mock reference',
                          dataset_release='1.0')


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded dataset dir. Should contain the classes folders.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        help="Location to extract.",
    )
    parser.add_argument(
        "-t",
        "--train_test_split",
        type=float,
        help="Train-test split ratio (e.g., 0.8 for 80% training and 20% testing).",
        default=0.8,
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=111, help="nnU-Net Dataset ID, default: 111"
    )
    args = parser.parse_args()
    
    
    base_dir = Path(os.path.abspath(args.input_folder))
    target_data_folder = Path(args.output_folder)
    os.makedirs(target_data_folder, exist_ok=True)
    
    print("[Dataset conversion] Converting...")
    convert_to_nnunet(
        src_data_folder=base_dir,
        target_data_folder=target_data_folder,
        train_test_split=args.train_test_split,
    )
    print("[Dataset conversion] Done!")

