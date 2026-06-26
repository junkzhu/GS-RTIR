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
    parser.add_argument("--max_workers", type=int, default=8, help="The maximum number of workers for parallel processing.")

    #-------------- dataset config --------------
    parser.add_argument("--dataset_type", default='TensoIR', help="The name of the dataset.")
    parser.add_argument("--dataset_name", help="The name of the scene.")
    parser.add_argument("--dataset_path", help="The path to the dataset directory.")
    parser.add_argument("--output_root", default=None, help="Override output base dir (e.g. run dir from run_all.py). Output will be output_root/dataset_name.")
    parser.add_argument("--diffusion_model", default="rgb2x", help="The name of the diffusion model, such as rgb2x, genprior.")
    
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size.")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle the dataset.")
    
    #-------------- optimization config --------------
    parser.add_argument("--ply_path", help="The path to the input .ply file.")
    
    #-------------- integrator config --------------
    parser.add_argument("--integrator_type", default='gsprim_prb', help="The type of integrator.")
    parser.add_argument("--geometry_threshold", type=float, default=0.3, help="The geometry threshold.")

    parser.add_argument("--selfocc_offset_max", type=float, default=0.1, help="The maximum self-occlusion offset.")
    parser.add_argument("--selfocc_mode", default='normal', choices=['normal', 'fixed'], help="Self-occlusion strategy: normal (normal-based + is_self_occlusion) or fixed (ray origin offset by constant).")
    parser.add_argument("--selfocc_offset_fixed", type=float, default=0.1, help="Fixed ray origin offset when selfocc_mode is 'fixed'.")

    parser.add_argument("--use_mis", type=str2bool, default=True, help="Enable MIS.")
    parser.add_argument("--max_bounce_num", type=int, default=3, help="The maximum number of bounces.")
    
    parser.add_argument("--dash_reso_sche", type=str2bool, default=False, help="Resolution scheduler described in DashGaussian.")

    parser.add_argument("--training_spp", type=int, default=16, help="The number of samples per pixel for training.")
    parser.add_argument("--primal_spp_mult", type=int, default=4, help="The multiplier for primal samples per pixel.")
    parser.add_argument("--spp_pt_rate", type=float, default=1.0, help="The rate of ray tracing point samples per pixel.")

    parser.add_argument("--separate_direct_indirect", type=str2bool, default=True, help="Separate output direct and indirect illumination.")
    
    parser.add_argument("--reset_attribute", type=str2bool, default=True, help="Reset attribute of albedo, roughness.")

    #-------------- emitter config --------------
    parser.add_argument("--hide_emitter", type=str2bool, default=True, help="Hide emitters.")
    
    parser.add_argument("--envmap_optimization", type=str2bool, default=True, help="Enable environment map optimization.")
    parser.add_argument("--envmap_path", default="/path/to/datasets/TensoIR/Environment_Maps/high_res_envmaps_1k/sunset.hdr", help="The path to the environment map.")
    
    parser.add_argument("--spherical_gaussian", type=str2bool, default=False, help="Enable spherical gaussian.")
    parser.add_argument("--num_sgs", type=int, default=24, help="The number of spherical gaussians.")

    #-------------- render & metrics --------------
    parser.add_argument("--render_spp", type=int, default=128, help="The number of samples per pixel for rendering.")
    parser.add_argument("--envmap_init_path", help="The path to the initial environment map npy.")
    parser.add_argument("--stride", type=int, default=1, help="The stride for rendering.")
    parser.add_argument("--only_relight", action="store_true", help="Skip base rendering and only run relight (if --relight).")

    #-------------- relight --------------
    parser.add_argument("--relight", action="store_true", help="Whether to relight the scene.")
    parser.add_argument("--envmap_root", default="/path/to/datasets/TensoIR/Environment_Maps/high_res_envmaps_2k", help="The path to the environment map directory.")

    args = parser.parse_args()
    return args
    
args = get_args()

#-------------- optimizer params --------------=
OPTIMIZE_PARAMS = ['shape.data', 'shape.opacities', 'shape.normals', 'shape.albedos', 'shape.roughnesses', 'shape.metallic']

#-------------- upsample & save iter --------------=
TRAIN_UPSAMPLE_ITER = [200, 400, 600]
SAVE_ENVMAP_ITER = [200, 400, 600]
RENDER_UPSAMPLE_ITER = []

#--------------folder--------------
__SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

# Define paths but don't create directories immediately
OUTPUT = os.path.realpath(os.path.join(__SCRIPT_DIR, './outputs'))
if args.output_root:
    DATASET_TYPE_DIR = os.path.realpath(os.path.abspath(args.output_root))
else:
    DATASET_TYPE_DIR = os.path.realpath(os.path.join(OUTPUT, f'./{args.dataset_type}'))
#OUTPUT_DIR = os.path.realpath(os.path.join(DATASET_TYPE_DIR, f'./{args.dataset_name}_{timestamp}'))
OUTPUT_DIR = os.path.realpath(os.path.join(DATASET_TYPE_DIR, f'./{args.dataset_name}'))

OUTPUT_RGB_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './rgb'))
OUTPUT_GBUFFER_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './gbuffer'))
OUTPUT_ALBEDO_DIR = os.path.realpath(os.path.join(OUTPUT_GBUFFER_DIR, './albedo'))
OUTPUT_ROUGHNESS_DIR = os.path.realpath(os.path.join(OUTPUT_GBUFFER_DIR, './roughness'))
OUTPUT_METALLIC_DIR = os.path.realpath(os.path.join(OUTPUT_GBUFFER_DIR, './metallic'))
OUTPUT_DEPTH_DIR = os.path.realpath(os.path.join(OUTPUT_GBUFFER_DIR, './depth'))
OUTPUT_NORMAL_DIR = os.path.realpath(os.path.join(OUTPUT_GBUFFER_DIR, './normal'))
OUTPUT_DIRECT_LIGHT_DIR = os.path.realpath(os.path.join(OUTPUT_GBUFFER_DIR, './direct_light'))
OUTPUT_INDIRECT_LIGHT_DIR = os.path.realpath(os.path.join(OUTPUT_GBUFFER_DIR, './indirect_light'))
OUTPUT_PLY_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './ply'))
OUTPUT_RENDER_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './renders'))
OUTPUT_RELIGHT_DIR = os.path.realpath(os.path.join(OUTPUT_RENDER_DIR, './relight'))
OUTPUT_ENVMAP_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, './envmap'))
OUTPUT_HYBRID_DIR = os.path.realpath(os.path.join(OUTPUT, f'./hybrid_results/{timestamp}'))

def ensure_dir(directory):
    """Ensure a directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# Create a dictionary to map directory constants to their paths for easy access
directory_paths = {
    'OUTPUT': OUTPUT,
    'DATASET_TYPE_DIR': DATASET_TYPE_DIR,
    'OUTPUT_DIR': OUTPUT_DIR,
    'OUTPUT_RGB_DIR': OUTPUT_RGB_DIR,
    'OUTPUT_GBUFFER_DIR': OUTPUT_GBUFFER_DIR,
    'OUTPUT_ALBEDO_DIR': OUTPUT_ALBEDO_DIR,
    'OUTPUT_ROUGHNESS_DIR': OUTPUT_ROUGHNESS_DIR,
    'OUTPUT_METALLIC_DIR': OUTPUT_METALLIC_DIR,
    'OUTPUT_DEPTH_DIR': OUTPUT_DEPTH_DIR,
    'OUTPUT_NORMAL_DIR': OUTPUT_NORMAL_DIR,
    'OUTPUT_DIRECT_LIGHT_DIR': OUTPUT_DIRECT_LIGHT_DIR,
    'OUTPUT_INDIRECT_LIGHT_DIR': OUTPUT_INDIRECT_LIGHT_DIR,
    'OUTPUT_PLY_DIR': OUTPUT_PLY_DIR,
    'OUTPUT_RENDER_DIR': OUTPUT_RENDER_DIR,
    'OUTPUT_RELIGHT_DIR': OUTPUT_RELIGHT_DIR,
    'OUTPUT_ENVMAP_DIR': OUTPUT_ENVMAP_DIR
}


# Create output directories only when a concrete scene is provided.
# This avoids side effects when constants.py is imported only for argument snapshotting.
ensure_dir(OUTPUT)
if args.dataset_name and args.dataset_type:
    ensure_dir(DATASET_TYPE_DIR)
    ensure_dir(OUTPUT_DIR)


del __SCRIPT_DIR
