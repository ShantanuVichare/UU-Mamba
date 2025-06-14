#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
join = os.path.join
expanduser = os.path.expanduser
"""
Please make sure your data is organized as follows:

data/  
├── nnUNet_raw/
│   ├── Dataset027_ACDC/
│   │   ├── imagesTr
│   │   │   ├── patient001_frame01_0000.nii.gz
│   │   │   ├── patient001_frame12_0000.nii.gz
│   │   │   ├── ...
│   │   ├── imagesTs
│   │   │   ├── patient101_frame01_0000.nii.gz
│   │   │   ├── patient101_frame14_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── patient001_frame01.nii.gz
│   │   │   ├── patient001_frame12.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTs
│   │   │   ├── patient101_frame01.nii.gz
│   │   │   ├── patient101_frame14.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json
│   ├── ...
"""
base = join(os.sep.join(__file__.split(os.sep)[:-3]), 'data') 
# or you can set your own path, e.g., base = '/home/user_name/Documents/UU-Mamba/data'
# base = '/workspace/UU-Mamba/data'
base = expanduser('~/Projects/uumamba/code/data')
nnUNet_raw = join(base, 'nnUNet_raw') # os.environ.get('nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # os.environ.get('nnUNet_preprocessed')
# nnUNet_results = join(base, 'nnUNet_results') # os.environ.get('nnUNet_results')

nnUNet_results_base = os.getenv('OUTPUT_PATH', expanduser('~/Projects/uumamba/code/results'))
nnUNet_results = join(nnUNet_results_base, 'nnUNet_results', 'running')
os.makedirs(nnUNet_results, exist_ok=True)
nnUNet_results_backup = join(nnUNet_results_base, 'nnUNet_results', 'running_backup')
# nnUNet_results_backup = os.path.expanduser('~/staging')
os.makedirs(nnUNet_results_backup, exist_ok=True)

if nnUNet_raw is None:
    print("nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnUNet_preprocessed is None:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnUNet_results is None:
    print("nnUNet_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")