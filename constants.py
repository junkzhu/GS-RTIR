"""This file stores the global constants used throughout the code base. Most importantly,
it specifies output paths and scene directory."""
import argparse
import os
import time
timestamp = time.strftime("%d%m_%H%M%S")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser()

    #-------------- dataset config --------------
    parser.add_argument("--dataset_type", help="The name of the dataset.")
    parser.add_argument("--dataset_name", help="The name of the scene.")
    parser.add_argument("--dataset_path", help="The path to the dataset directory.")
    
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size.")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle the dataset.")
    
    #-------------- optimization config --------------
    parser.add_argument("--ply_path", help="The path to the input .ply file.")
    parser.add_argument("--refine_path", help="The path to the refined output .ply file.")
    
    #-------------- integrator config --------------
    parser.add_argument("--integrator_type", default='gsprim_prb', help="The type of integrator.")
    parser.add_argument("--geometry_threshold", type=float, default=0.3, help="The geometry threshold.")

    parser.add_argument("--selfocc_offset_max", type=float, default=0.1, help="The maximum self-occlusion offset.")
    parser.add_argument("--use_mis", type=str2bool, default=True, help="Enable MIS.")
    parser.add_argument("--max_bounce_num", type=int, default=4, help="The maximum number of bounces.")
    
    parser.add_argument("--training_spp", type=int, default=16, help="The number of samples per pixel for training.")
    parser.add_argument("--primal_spp_mult", type=int, default=4, help="The multiplier for primal samples per pixel.")
    parser.add_argument("--spp_pt_rate", type=float, default=1.0, help="The rate of ray tracing point samples per pixel.")

    parser.add_argument("--separate_direct_indirect", type=str2bool, default=False, help="Separate output direct and indirect illumination.")
    
    #-------------- refine config --------------
    parser.add_argument("--refine_spp", type=int, default=4, help="The number of samples per pixel for refining.")
    parser.add_argument("--refine_niter", type=int, default=500, help="The number of iterations for refining.")
    parser.add_argument("--reset_attribute", type=str2bool, default=True, help="Reset attribute of albedo, roughness.")

    #-------------- emitter config --------------
    parser.add_argument("--hide_emitter", type=str2bool, default=True, help="Hide emitters.")
    
    parser.add_argument("--envmap_optimization", type=str2bool, default=True, help="Enable environment map optimization.")
    parser.add_argument("--envmap_path", default="/home/zjk/datasets/TensoIR/Environment_Maps/high_res_envmaps_1k/sunset.hdr", help="The path to the environment map.")
    
    parser.add_argument("--spherical_gaussian", type=str2bool, default=True, help="Enable spherical gaussian.")
    parser.add_argument("--num_sgs", type=int, default=16, help="The number of spherical gaussians.")

    #-------------- render & metrics --------------
    parser.add_argument("--render_spp", type=int, default=128, help="The number of samples per pixel for rendering.")
    parser.add_argument("--envmap_init_path", help="The path to the initial environment map npy.")

    #-------------- relight --------------
    parser.add_argument("--relight", action="store_true", help="Whether to relight the scene.")
    parser.add_argument("--envmap_root", default="/home/zjk/datasets/TensoIR/Environment_Maps/high_res_envmaps_2k", help="The path to the environment map directory.")

    args = parser.parse_args()
    return args
    
args = get_args()

#-------------- optimizer params --------------=
OPTIMIZE_PARAMS = ['shape.data', 'shape.opacities', 'shape.normals', 'shape.albedos', 'shape.roughnesses']
REFINE_PARAMS = ['shape.data', 'shape.opacities', 'shape.normals', 'shape.sh_coeffs']

#-------------- upsample & save iter --------------=
REFINE_UPSAMPLE_ITER = []
TRAIN_UPSAMPLE_ITER = [64, 128, 256]
SAVE_ENVMAP_ITER = [64, 128, 256]
RENDER_UPSAMPLE_ITER = []

#--------------folder--------------
__SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

OUTPUT = os.path.realpath(os.path.join(__SCRIPT_DIR, './outputs'))
os.makedirs(OUTPUT, exist_ok=True)

DATASET_TYPE_DIR = os.path.realpath(os.path.join(OUTPUT, f'./{args.dataset_type}'))
os.makedirs(DATASET_TYPE_DIR, exist_ok=True)

#OUTPUT_DIR = os.path.realpath(os.path.join(DATASET_TYPE_DIR, f'./{args.dataset_name}_{timestamp}'))
OUTPUT_DIR = os.path.realpath(os.path.join(DATASET_TYPE_DIR, f'./{args.dataset_name}'))
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

OUTPUT_RELIGHT_DIR = os.path.realpath(os.path.join(OUTPUT_RENDER_DIR, './relight'))
os.makedirs(OUTPUT_RELIGHT_DIR, exist_ok=True)

OUTPUT_ENVMAP_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './envmap'))
os.makedirs(OUTPUT_ENVMAP_DIR, exist_ok=True)

del __SCRIPT_DIR