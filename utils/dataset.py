import os
import torch
import torch.nn.functional as F

import nibabel as nib
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold


# Dataset for 3D MRI images
class TumorMRIDataset(Dataset):
    def __init__(self, root_dir, mode='nib', modalities = ['t1ce', 't1', 't2', 'flair'], segmentation=False, limit=None, target_shape=None):
        self.mode = mode
        self.root_dir = root_dir
        self.modalities = modalities
        self.samples = self._load_samples(root_dir, limit)
        self.target_shape = target_shape
        self.segmentation = 'seg' if segmentation else None
        self.segmentation_targets = self._load_segmentation_targets(root_dir, limit)
        self.cache_slice_meta = {
            'chunk_idx': -1,
        }
        self.data_shape = (155, 240, 240)

    def _load_samples(self, root_dir, limit=None):
        samples = []
        for label in ['HGG', 'LGG']:
            label_specific_sample_count = 0
            folder_path = os.path.join(root_dir, label)
            for patient_folder in os.listdir(folder_path):
                file_paths = [None] * len(self.modalities)
                for i, modality in enumerate(self.modalities):
                    file_paths[i] = os.path.join(folder_path, patient_folder, f'{patient_folder}_{modality}.nii')
                samples.append((file_paths, 0 if label == 'HGG' else 1))
                label_specific_sample_count += len(file_paths)
                
                if limit is not None and label_specific_sample_count >= limit:
                    break
        return samples

    def _load_segmentation_targets(self, root_dir, limit=None):
        if self.segmentation is None:
            return None
        segmentation_targets = []
        for label in ['HGG', 'LGG']:
            folder_path = os.path.join(root_dir, label)
            for patient_folder in os.listdir(folder_path):
                seg_file_path = os.path.join(folder_path, patient_folder, f'{patient_folder}_{self.segmentation}.nii')
                segmentation_targets.append(seg_file_path)
                if limit is not None and len(segmentation_targets) >= limit:
                    break
        return segmentation_targets

    def __len__(self):
        # if self.segmentation is not None:
        #     # For segmentation, we will split tensor into chunk of slices
        #     return len(self.samples) * self.data_shape[0]//20
        return len(self.samples)

    def get_cached_slices(self, idx):
        # Load the slices from the cache for segmentation
        chunk_idx = idx // (self.data_shape[0]//20)
        slice_idx = idx % (self.data_shape[0]//20)
        # print('index:', idx, chunk_idx, slice_idx)
        if (self.cache_slice_meta['chunk_idx'] != chunk_idx):
            self.cache_slice_meta['chunk_idx'] = chunk_idx
            
            self.cache_slice_meta['data'] = self.get_nib(chunk_idx)
            
        data_tensors, seg_tensors = self.cache_slice_meta['data']
        start_idx = slice_idx * 20
        end_idx = start_idx + 20
        return data_tensors[:, start_idx:end_idx, :, :], seg_tensors[:, start_idx:end_idx, :, :]            
        

    def __getitem__(self, idx):
        if self.mode == 'nib':
            # if self.segmentation is not None:
            #     return self.get_cached_slices(idx)
            return self.get_nib(idx)
        elif self.mode == 'pt':
            return self.get_pt(idx)
        else:
            raise ValueError('Invalid mode')
        
    def get_pt(self, idx):
        pre = 1 if idx>258 else 0
        # print(pre, idx)
        file_path = f'./test/pt_export/{pre}_{idx}.pt'
        return torch.load(file_path), torch.tensor(pre, dtype=torch.long)
        
    def get_nib(self, idx):
        file_paths, label = self.samples[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        data_tensors = []
        for file_path in file_paths:
            img = nib.load(file_path).get_fdata()
            img = self._pad_or_crop(img)
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
            data_tensors.append(img_tensor)
        data_tensors = torch.stack(data_tensors)
        # data_tensors = F.interpolate(data_tensors, size=(256,224), mode='bicubic', align_corners=False)
        
        if self.segmentation is not None:
            seg_file_path = self.segmentation_targets[idx]
            seg_img = nib.load(seg_file_path).get_fdata()
            seg_img = self._pad_or_crop(seg_img)
            seg_tensor = torch.tensor(seg_img, dtype=torch.float32).permute(2,0,1)
            
            seg_new = torch.zeros_like(seg_tensor)
            seg_new[seg_tensor == 4] = 3
            seg_new[seg_tensor == 2] = 1
            seg_new[seg_tensor == 1] = 2
            seg_tensors = torch.stack([seg_new])
            # seg_tensors = F.interpolate(seg_tensors, size=(256,224), mode='bicubic', align_corners=False)
            return data_tensors, seg_tensors
        
        return data_tensors, label

    def _pad_or_crop(self, img):
        target_shape = self.target_shape if self.target_shape else img.shape
        pad_size = [(max(0, target - img_dim)) for target, img_dim in zip(target_shape, img.shape)]
        pad_widths = [(p // 2, p - p // 2) for p in pad_size]
        img_padded = np.pad(img, pad_widths, mode='constant', constant_values=0)
        return img_padded[:target_shape[0], :target_shape[1], :target_shape[2]]


# Split the dataset into train and test sets by class
def split_dataset_by_class(dataset, train_ratio=0.8):
    HGG_samples = [sample for sample in dataset.samples if sample[1] == 0]
    LGG_samples = [sample for sample in dataset.samples if sample[1] == 1]
    

    # Split each class
    HGG_train, HGG_test = train_test_split(HGG_samples, train_size=train_ratio, shuffle=True)
    LGG_train, LGG_test = train_test_split(LGG_samples, train_size=train_ratio, shuffle=True)

    # Combine the train and test samples
    train_samples = HGG_train + LGG_train
    test_samples = HGG_test + LGG_test
    distribution_info = {
        'HGG_train': len(HGG_train),
        'HGG_test': len(HGG_test),
        'LGG_train': len(LGG_train),
        'LGG_test': len(LGG_test),
    }

    return train_samples, test_samples, distribution_info


def split_dataset_into_loaders(dataset, train_ratio=0.8, batch_size=32, random_seed=42):
    """Splits a dataset into training and testing sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): The ratio of the training set size to the whole dataset size.
        batch_size (int): The batch size for the DataLoaders.
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training and testing DataLoaders.
    """
    num_train = int(len(dataset) * train_ratio)
    num_test = len(dataset) - num_train

    train_dataset, test_dataset = random_split(dataset, 
                                                [num_train, num_test],
                                                generator=torch.Generator().manual_seed(random_seed))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    return train_loader, test_loader
