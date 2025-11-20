"""This file stores the global constants used throughout the code base. Most importantly,
it specifies output paths and scene directory."""

import os
#--------------dataset--------------
DATASET_PATH = 'D:/ZJK/hotdog_aov'

PLY_PATH = 'D:/ZJK/GS-RTIR/outputs/ply/refined.ply'
ENVMAP_PATH = 'D:/dataset/Environment_Maps/high_res_envmaps_1k/sunset.hdr'

REFINE_PATH = 'D:/ZJK/GS-RTIR/outputs/ply/point_refined.ply'

RESET_ATTRIBUTE = False

#--------------integrator--------------
#INTEGRATOR = 'volprim_rf_basic'
INTEGRATOR = 'gsprim_prb'
MAX_BOUNCE_NUM = 4

SPP = 8
SPP_PT_RATE = 1.0 # not necessary

PRIMAL_SPP_MULT = 4
USE_MIS = True

#--------------emitter--------------
OPTIMIZE_ENVMAP = False


#--------------training--------------
BATCH_SIZE = 6
NITER = 512
OPTIMIZE_PARAMS = ['shape.normals','shape.albedos','shape.roughnesses']
TRAIN_UPSAMPLE_ITER = [64, 128, 256]

REFINE_NITER = 128
REFINE_PARAMS = ['shape.data', 'shape.opacities', 'shape.normals', 'shape.sh_coeffs']
REFINE_UPSAMPLE_ITER = [32, 64]

#--------------render & metrics--------------
RENDER_SPP = 128
RENDER_UPSAMPLE_ITER = []

#--------------folder--------------
__SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

OUTPUT_DIR = os.path.realpath(os.path.join(__SCRIPT_DIR, './outputs'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_REFINE_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './refine'))
os.makedirs(OUTPUT_REFINE_DIR, exist_ok=True)

OUTPUT_OPT_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './opt'))
os.makedirs(OUTPUT_OPT_DIR, exist_ok=True)

OUTPUT_EXTRA_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './extra'))
os.makedirs(OUTPUT_EXTRA_DIR, exist_ok=True)

OUTPUT_PLY_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './ply'))
os.makedirs(OUTPUT_PLY_DIR, exist_ok=True)

OUTPUT_RENDER_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './renders'))
os.makedirs(OUTPUT_RENDER_DIR, exist_ok=True)

OUTPUT_ENVMAP_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './envmap'))
os.makedirs(OUTPUT_ENVMAP_DIR, exist_ok=True)

del __SCRIPT_DIR