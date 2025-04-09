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


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg


def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


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


def convert_brats19_to_nnunet(src_train_dir: Path, src_test_dir: Path, target_data_folder: Path):
        
    out_dir, train_dir, trainlabels_dir, test_dir, testlabels_dir = make_out_dirs(target_data_folder)

    # setting up nnU-Net folders
    def transfer_files(brats_data_dir, imagestr, labelstr):
        # print(f"Transferring files from {brats_data_dir} to {imagestr} and {labelstr}")
        # print(f"File count: {len(os.listdir(brats_data_dir))}")
        # case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)
        case_ids = [c for c in os.listdir(brats_data_dir) if c.startswith('BraTS')]
        for c in case_ids:
            shutil.copy(join(brats_data_dir, c, c + "_t1.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
            shutil.copy(join(brats_data_dir, c, c + "_t1ce.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
            shutil.copy(join(brats_data_dir, c, c + "_t2.nii.gz"), join(imagestr, c + '_0002.nii.gz'))
            shutil.copy(join(brats_data_dir, c, c + "_flair.nii.gz"), join(imagestr, c + '_0003.nii.gz'))

            copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir, c, c + "_seg.nii.gz"),
                                                                join(labelstr, c + '.nii.gz'))
        return len(case_ids)

    num_training_cases = transfer_files(src_train_dir, train_dir, trainlabels_dir)
    num_testing_cases = transfer_files(src_test_dir, test_dir, testlabels_dir)

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
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')

def create_train_test_split_dirs(base_dir: Path, sub_dirs = ['LGG', 'HGG'], train_test_split = 0.8):

    train_dir = base_dir / 'training'
    test_dir = base_dir / 'testing'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    '''
    Need to copy/link split-ratio files from base_dir+sub_dir to train_dir and remaining to test_dir
    '''

    train_paths = []
    test_paths = []
    sub_dir_map = {}
    for sub_dir in sub_dirs:
        file_paths = []
        for filepath in os.listdir(base_dir / sub_dir):
            if filepath.startswith('BraTS'):
                full_path = Path(os.path.abspath(base_dir / sub_dir / filepath))
                file_paths.append(full_path)
                sub_dir_map[str(full_path)] = sub_dir
        random.shuffle(file_paths)
        train_size = int(len(file_paths) * train_test_split)
        train_paths.extend(file_paths[:train_size])
        test_paths.extend(file_paths[train_size:])
        print(f"Subdir: {sub_dir}, Total: {len(file_paths)}, Train: {len(file_paths[:train_size])}, Test: {len(file_paths[train_size:])}")
    

    compression_start_time = time.time()
    def compress_file(src_file, target_file):
        with open(src_file, 'rb') as f_in:
            with gzip.open(target_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    for src_paths, target_base_path in [(train_paths, train_dir), (test_paths, test_dir)]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for src_path in src_paths:
                target_path = target_base_path / os.path.basename(src_path)
                os.makedirs(target_path, exist_ok=True)
                # open(target_path / sub_dir, 'w').close()  # Create empty file to store sub_dir (class) name
                for src_file_name in os.listdir(src_path):
                    src_file = src_path / src_file_name
                    target_file = target_path / (src_file_name + '.gz')
                    futures.append(executor.submit(compress_file, src_file, target_file))
            # Wait for all compression tasks to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()

    print(f"Time taken for compression: {(time.time() - compression_start_time):.3f} seconds")
    
    print(f"Generated {len(train_paths)} cases to training and {len(test_paths)} cases to testing.")
    return train_dir, test_dir, sub_dir_map

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
        "-c",
        "--classes",
        type=str,
        help="Comma-separated list of input classes directories in Input folder (e.g., 'LGG,HGG').",
        default='LGG,HGG',
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
    train_dir, test_dir, sub_dir_map = create_train_test_split_dirs(base_dir, sub_dirs=args.classes.split(','), train_test_split=args.train_test_split)
    target_data_folder = Path(args.output_folder)
    os.makedirs(target_data_folder, exist_ok=True)
    
    # Store the sub_dir_map in the target_data_folder
    with open(target_data_folder / 'sub_dir_map.json', 'w') as f:
        json.dump(sub_dir_map, f)
    
    print("Converting...")
    convert_brats19_to_nnunet(
        src_train_dir=train_dir,
        src_test_dir=test_dir,
        target_data_folder=target_data_folder,
    )
    print("Done!")

