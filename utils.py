# =========================================================================
#   (c) Copyright 2022
#   All rights reserved
#   Programs written by Haodi Jiang
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================


import os
import sys
from os import path
from pathlib import Path

def create_predict_default_dirs():
    if (len(sys.argv)) > 1:
        if ('-p' in sys.argv or '--path' in sys.argv) and 'model' in sys.argv:
            Path('model').mkdir(exist_ok=True)
        if ('-d' in sys.argv or '--data' in sys.argv) and 'test_data' in sys.argv:
            Path('test_data').mkdir(exist_ok=True)
        if ('-r' in sys.argv or '--result_dir' in sys.argv) and 'result' in sys.argv:
            Path('result' + os.sep + 'bx').mkdir(parents=True, exist_ok=True)
            Path('result' + os.sep + 'by').mkdir(parents=True,exist_ok=True)
    else:
        Path('model').mkdir(exist_ok=True)
        Path('test_data').mkdir(exist_ok=True)
        Path('result').mkdir(parents=True, exist_ok=True)
        # Path('result' + os.sep + 'bx').mkdir(parents=True, exist_ok=True)
        # Path('result' + os.sep + 'by').mkdir(parents=True,exist_ok=True)

def create_training_default_dirs():
    if (len(sys.argv)) > 1:
        if ('-p' in sys.argv or '--path' in sys.argv) and 'model' in sys.argv:
            Path('model').mkdir(exist_ok=True)
        if ('-d' in sys.argv or '--data' in sys.argv) and 'test_data' in sys.argv:
            Path('test_data').mkdir(exist_ok=True)
        if ('-r' in sys.argv or '--result_dir' in sys.argv) and 'result' in sys.argv:
            Path('result').mkdir(parents=True, exist_ok=True)
            # Path('result' + os.sep + 'bx').mkdir(parents=True, exist_ok=True)
            # Path('result' + os.sep + 'by').mkdir(parents=True,exist_ok=True)
    else:
        Path('model').mkdir(exist_ok=True)
        Path('test_data').mkdir(exist_ok=True)
        Path('result').mkdir(parents=True, exist_ok=True)
        # Path('result' + os.sep + 'bx').mkdir(parents=True, exist_ok=True)
        # Path('result' + os.sep + 'by').mkdir(parents=True,exist_ok=True)


def create_default_dirs_training():
    if (len(sys.argv)) > 1:
        if ('-p' in sys.argv or '--path' in sys.argv) and 'model' in sys.argv:
            Path('model').mkdir(exist_ok=True)
        if ('-t' in sys.argv or '--train' in sys.argv) and 'train_data' in sys.argv:
            Path('train_data').mkdir(exist_ok=True)
        if ('-v' in sys.argv or '--val' in sys.argv) and 'val_data' in sys.argv:
            Path('val_data').mkdir(exist_ok=True)

    else:
        Path('model').mkdir(exist_ok=True)
        Path('train_data').mkdir(exist_ok=True)
        Path('val_data').mkdir(parents=True, exist_ok=True)
