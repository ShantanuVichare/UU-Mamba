import os
import sys
import shutil
from datetime import datetime
from time import time, sleep

from nnunetv2.paths import nnUNet_results_backup

def getTimeString(timestamp=None):
    if timestamp is not None:
        return datetime.fromtimestamp(timestamp).strftime("%y%m%d-%Hh%Mm")
    return datetime.now().strftime("%y%m%d-%Hh%Mm")

assert len(sys.argv) == 2, "This script takes the time in format YYMMDD-HHhMMm as an argument."

scriptEndTime = datetime.strptime(sys.argv[1], "%y%m%d-%Hh%Mm").timestamp()
delayTime = scriptEndTime - time()
if delayTime < 0:
    print("The specified time is in the past. Exiting.")
    exit()
print(f'Starting at {getTimeString()}. Will start transfer at {getTimeString(scriptEndTime)}', flush=True)

sleep(delayTime)

# Compress self.output_folder to {nnUNet_results_backup}/results_backup.tar.gz
print(f'Backing up results... at {getTimeString()}')

shutil.make_archive(os.path.join(nnUNet_results_backup, 'results_backup_MANUAL'), 'gztar', os.getenv('OUTPUT_PATH'))
print('Backup complete.')
