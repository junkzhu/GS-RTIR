"""This file stores the global constants used throughout the code base. Most importantly,
it specifies output paths and scene directory."""

import os
#--------------dataset--------------
DATASET_PATH = 'D:/dataset/hotdog'

PLY_PATH = 'D:/dataset/hotdog/point_cloud.ply'
ENVMAP_PATH = 'D:/dataset/sunset.hdr'


#--------------integrator--------------
#INTEGRATOR = 'volprim_rf_basic'
INTEGRATOR = 'gsprim_rf'
SPP = 32


#--------------training--------------
BATCH_SIZE = 6
NITER = 200
OPTIMIZE_PARAMS = ['shape.sh_coeffs']
RENDER_UPSAMPLE_ITER = [64, 128, 256]


#--------------folder--------------
__SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

OUTPUT_DIR = os.path.realpath(os.path.join(__SCRIPT_DIR, './outputs'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_OPT_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './opt'))
os.makedirs(OUTPUT_OPT_DIR, exist_ok=True)

OUTPUT_EXTRA_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './extra'))
os.makedirs(OUTPUT_EXTRA_DIR, exist_ok=True)

del __SCRIPT_DIR