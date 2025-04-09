import importlib
import pkgutil

from batchgenerators.utilities.file_and_folder_operations import *


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    DEBUG = False
    if DEBUG: print(f"Searching in folder: {folder}, module: {current_module}")
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        if DEBUG: print(f"Checking module: {modname}, is package: {ispkg}")
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                if DEBUG: print(f"Found class {class_name} in module: {modname}")
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                if DEBUG: print(f"Recursively searching in package: {modname}")
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break

    if tr is not None:
        if DEBUG: print(f"Selected Class {class_name} found in folder: {folder}")
    return tr