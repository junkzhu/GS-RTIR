"""This file stores the global constants used throughout the code base. Most importantly,
it specifies output paths and scene directory."""
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", help="Synthetic4Relight")
    parser.add_argument("--dataset_name", help="chair")
    parser.add_argument("--dataset_path", help="E:/dataset/Synthetic4Relight/chair")

    parser.add_argument("--ply_path", help="E:/dataset/Synthetic4Relight/chair.ply")
    parser.add_argument("--refine_path", help="E:/dataset/Synthetic4Relight/chair_refined.ply")

    #--------------render & metrics--------------
    parser.add_argument("--render_spp", type=int, default=128, help="number of samples per pixel for rendering")

    args = parser.parse_args()
    return args
    
args = get_args()

#--------------dataset--------------
DATASET_TYPE = args.dataset_type
DATASET_NAME = args.dataset_name
DATASET_PATH = args.dataset_path

PLY_PATH = args.ply_path
ENVMAP_PATH = 'D:/dataset/Environment_Maps/high_res_envmaps_1k/sunset.hdr'

REFINE_PATH = args.refine_path

RESET_ATTRIBUTE = False

#--------------integrator--------------
#INTEGRATOR = 'volprim_rf_basic'
INTEGRATOR = 'gsprim_prb'
MAX_BOUNCE_NUM = 4

SPP = 8
SPP_PT_RATE = 1.0 # not necessary

PRIMAL_SPP_MULT = 4
USE_MIS = True
HIDE_EMITTER = True

#--------------emitter--------------
OPTIMIZE_ENVMAP = False

SPHERICAL_GAUSSIAN = True
NUM_SGS = 16

#--------------training--------------
BATCH_SIZE = 6
NITER = 512
OPTIMIZE_PARAMS = ['shape.normals','shape.albedos','shape.roughnesses']
TRAIN_UPSAMPLE_ITER = [64, 128, 256]

REFINE_NITER = 128
REFINE_PARAMS = ['shape.data', 'shape.opacities', 'shape.normals', 'shape.sh_coeffs']
REFINE_UPSAMPLE_ITER = [32, 64]

#--------------render & metrics--------------
RENDER_SPP = args.render_spp
RENDER_UPSAMPLE_ITER = []

#--------------folder--------------
__SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

OUTPUT = os.path.realpath(os.path.join(__SCRIPT_DIR, './outputs'))
os.makedirs(OUTPUT, exist_ok=True)

OUTPUT_DIR = os.path.realpath(os.path.join(OUTPUT, f'./{DATASET_TYPE}_{DATASET_NAME}'))
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