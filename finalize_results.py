
import os
join = os.path.join
dirname = os.path.dirname

import datetime
import shutil

from nnunetv2.paths import nnUNet_results

def getTime():
    return datetime.datetime.now().strftime("%y%m%d-%Hh%Mm")


startTime = getTime()
run_id = os.getenv('RUN_ID', startTime)
shutil.move(nnUNet_results, join(dirname(nnUNet_results), run_id) )
print(f"[Finalize] Results moved from '{nnUNet_results}' to '{join(dirname(nnUNet_results), run_id)}'")
