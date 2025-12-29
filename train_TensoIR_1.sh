#!/bin/bash

# Example commands for direct execution (TensoIR dataset)
# CUDA_VISIBLE_DEVICES=0 python refine/refine_TensoIR.py --dataset_type TensoIR --dataset_name lego --dataset_path /home/zjk/datasets/TensoIR/lego --ply_path /home/zjk/datasets/TensoIR/3dgrt/lego.pt --refine_path /home/zjk/datasets/TensoIR/3dgrt/lego_refined.ply
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type TensoIR --dataset_name lego --dataset_path /home/zjk/datasets/TensoIR/lego --ply_path /home/zjk/datasets/TensoIR/3dgrt/lego_refined.ply --selfocc_offset_max 0.1 --geometry_threshold 0.3
# CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type TensoIR --dataset_name lego --dataset_path /home/zjk/datasets/TensoIR/lego --ply_path /home/zjk/code/GS-RTIR/outputs/TensoIR/lego/ply/iter_299.ply --render_spp 64 --envmap_init_path /home/zjk/code/GS-RTIR/outputs/TensoIR/lego/envmap/optimized_sgs_0299.npy --relight --envmap_root /home/zjk/datasets/TensoIR/Environment_Maps --selfocc_offset_max 0.1 --geometry_threshold 0.3
# CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type TensoIR --dataset_name lego --dataset_path /home/zjk/datasets/TensoIR/lego --envmap_root /home/zjk/datasets/TensoIR/Environment_Maps --relight

# Configuration variables
CUDA_DEVICE="2"
DATASET_TYPE="TensoIR"
DATASET_ROOT="/home/zjk/datasets/TensoIR"
OUTPUT_ROOT="/home/zjk/code/GS-RTIR/outputs/TensoIR"
ENVIRONMENT_MAPS="/home/zjk/datasets/TensoIR/Environment_Maps"
ITERATION="299"  # Default iteration number, can be overridden by command line argument

# Enable/disable switches for each step
enable_refine=true
enable_train=true
enable_render=true
enable_metrics=true
enable_relight=false  # Switch for relight functionality

# Scene-specific parameters
declare -A SCENE_PARAMS
SCENE_PARAMS[lego,offset]="0.15"
SCENE_PARAMS[lego,geometry_threshold]="0.3"

SCENE_PARAMS[hotdog,offset]="0.05"
SCENE_PARAMS[hotdog,geometry_threshold]="0.01"

SCENE_PARAMS[armadillo,offset]="0.1"
SCENE_PARAMS[armadillo,geometry_threshold]="0.3"

SCENE_PARAMS[ficus,offset]="0.25"
SCENE_PARAMS[ficus,geometry_threshold]="0.25"

# Function to run refine for a scene
run_refine() {
    local scene=$1
    # Extract base scene name (e.g., "lego" from "lego2")
    local base_scene=$(echo $scene | sed 's/[0-9]*$//')
    echo "Running refine for $scene..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python refine/refine_TensoIR.py \
        --dataset_type $DATASET_TYPE \
        --dataset_name $scene \
        --dataset_path $DATASET_ROOT/$base_scene \
        --ply_path $DATASET_ROOT/3dgrt/${base_scene}.pt \
        --refine_path $DATASET_ROOT/3dgrt/${scene}_refined.ply
}

# Function to run train for a scene
run_train() {
    local scene=$1
    # Extract base scene name (e.g., "lego" from "lego2")
    local base_scene=$(echo $scene | sed 's/[0-9]*$//')
    # Use base scene parameters if available, otherwise use defaults
    local offset=${SCENE_PARAMS[$base_scene,offset]:-"0.1"}
    local geometry_threshold=${SCENE_PARAMS[$base_scene,geometry_threshold]:-"0.3"}
    echo "Running train for $scene (base: $base_scene)..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py \
        --dataset_type $DATASET_TYPE \
        --dataset_name $scene \
        --dataset_path $DATASET_ROOT/$base_scene \
        --ply_path $DATASET_ROOT/3dgrt/${base_scene}_refined.ply \
        --selfocc_offset_max $offset \
        --geometry_threshold $geometry_threshold
}

# Function to run render for a scene
run_render() {
    local scene=$1
    # Extract base scene name (e.g., "lego" from "lego2")
    local base_scene=$(echo $scene | sed 's/[0-9]*$//')
    # Use base scene parameters if available, otherwise use defaults
    local offset=${SCENE_PARAMS[$base_scene,offset]:-"0.1"}
    local geometry_threshold=${SCENE_PARAMS[$base_scene,geometry_threshold]:-"0.3"}
    local iter_padded=$(printf "%04d" $ITERATION)
    echo "Running render for $scene with iteration $ITERATION..."
    
    # Build render command with optional relight flag
    local render_cmd="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python render.py \
        --dataset_type $DATASET_TYPE \
        --dataset_name $scene \
        --dataset_path $DATASET_ROOT/$base_scene \
        --ply_path $OUTPUT_ROOT/$scene/ply/iter_${ITERATION}.ply \
        --render_spp 64 \
        --envmap_init_path $OUTPUT_ROOT/$scene/envmap/optimized_sgs_${iter_padded}.npy"
    
    if [ "$enable_relight" = true ]; then
        render_cmd="$render_cmd \
        --relight"
    fi
    
    render_cmd="$render_cmd \
        --selfocc_offset_max $offset \
        --geometry_threshold $geometry_threshold"
    
    # Execute the command
    eval $render_cmd
}

# Function to run metrics for a scene
run_metrics() {
    local scene=$1
    # Extract base scene name (e.g., "lego" from "lego2")
    local base_scene=$(echo $scene | sed 's/[0-9]*$//')
    echo "Running metrics for $scene..."
    
    # Build metrics command with optional relight flag
    local metrics_cmd="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python metrics.py \
        --dataset_type $DATASET_TYPE \
        --dataset_name $scene \
        --dataset_path $DATASET_ROOT/$base_scene"
    
    if [ "$enable_relight" = true ]; then
        metrics_cmd="$metrics_cmd \
        --relight"
    fi
    
    # Execute the command
    eval $metrics_cmd
}

# Function to run all steps for a scene
run_all() {
    local scene=$1
    echo "\n=== Processing scene: $scene ==="
    
    # Run steps based on enable switches
    if [ "$enable_refine" = true ]; then
        run_refine $scene
    else
        echo "Skipping refine step (disabled)..."
    fi
    
    if [ "$enable_train" = true ]; then
        run_train $scene
    else
        echo "Skipping train step (disabled)..."
    fi
    
    if [ "$enable_render" = true ]; then
        run_render $scene
    else
        echo "Skipping render step (disabled)..."
    fi
    
    if [ "$enable_metrics" = true ]; then
        run_metrics $scene
    else
        echo "Skipping metrics step (disabled)..."
    fi
    
    echo "=== Finished processing scene: $scene ===\n"
}

# Parse command line arguments
# Check if the first argument is a number (iteration)
if [[ "$1" =~ ^[0-9]+$ ]]; then
    ITERATION="$1"
    shift  # Remove the iteration argument from the list
fi

# Main execution
if [ $# -eq 0 ]; then
    # Run all scenes if no arguments provided
    echo "Running all scenes with iteration $ITERATION..."
    for scene in lego hotdog armadillo ficus; do
        run_all $scene
    done
else
    # Run only specified scenes
    echo "Running specified scenes: $@ with iteration $ITERATION..."
    for scene in "$@"; do
        run_all $scene
    done
fi

echo "Done!"