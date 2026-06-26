#!/bin/bash

# Example commands for direct execution (TensoIR dataset)
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type TensoIR --dataset_name lego --dataset_path /path/to/datasets/TensoIR/lego --ply_path /path/to/datasets/TensoIR/3dgrt/lego_refined.ply --selfocc_offset_max 0.5 --geometry_threshold 0.3
# CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type TensoIR --dataset_name lego --dataset_path /path/to/datasets/TensoIR/lego --ply_path /path/to/PTIR-Mitsuba/outputs/TensoIR/lego/ply/iter_299.ply --render_spp 64 --envmap_init_path /path/to/PTIR-Mitsuba/outputs/TensoIR/lego/envmap/optimized_sgs_0299.npy --relight --envmap_root /path/to/datasets/TensoIR/Environment_Maps --selfocc_offset_max 0.5 --geometry_threshold 0.3
# CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type TensoIR --dataset_name lego --dataset_path /path/to/datasets/TensoIR/lego --envmap_root /path/to/datasets/TensoIR/Environment_Maps --relight

# Configuration variables
CUDA_DEVICE="0"  # Default CUDA device, can be overridden by --cuda_device argument
DATASET_TYPE="TensoIR"
DATASET_ROOT="/path/to/datasets/TensoIR"
OUTPUT_ROOT="./outputs/TensoIR"
ENVIRONMENT_MAPS="/path/to/datasets/TensoIR/Environment_Maps"
ITERATION="799"  # Default iteration number, can be overridden by command line argument

# Enable/disable switches for each step
enable_train=true
enable_render=true
enable_metrics=true
enable_relight=false  # Switch for relight functionality
RESUME=false  # Resume training from latest checkpoint in output dir

# Scene-specific parameters
declare -A SCENE_PARAMS
SCENE_PARAMS[lego,offset]="0.1"
SCENE_PARAMS[lego,geometry_threshold]="0.5"

SCENE_PARAMS[hotdog,offset]="0.1"
SCENE_PARAMS[hotdog,geometry_threshold]="0.5"

SCENE_PARAMS[armadillo,offset]="0.1"
SCENE_PARAMS[armadillo,geometry_threshold]="0.5"

SCENE_PARAMS[ficus,offset]="0.1"
SCENE_PARAMS[ficus,geometry_threshold]="0.5"

# Scenes that belong to Synthetic4Relight (not TensoIR); hint if user passes them here
SYNTHETIC4RELIGHT_SCENES="air_baloons jugs chair"

# Function to run train for a scene
run_train() {
    local scene=$1
    for s in $SYNTHETIC4RELIGHT_SCENES; do
        if [ "$scene" = "$s" ]; then
            echo "Note: $scene is a Synthetic4Relight scene. Use: ./train_Synthetic4Relight.sh $scene --resume"
            break
        fi
    done
    # Extract base scene name (e.g., "lego" from "lego2")
    local base_scene=$(echo $scene | sed 's/[0-9]*$//')
    # Use base scene parameters if available, otherwise use defaults
    local offset=${SCENE_PARAMS[$base_scene,offset]:-"0.5"}
    local geometry_threshold=${SCENE_PARAMS[$base_scene,geometry_threshold]:-"0.3"}
    # When --resume: only pass --resume if this scene has a checkpoint (ply/iter_*.ply)
    local do_resume=false
    if [ "$RESUME" = true ]; then
        local ply_dir="./outputs/$DATASET_TYPE/$scene/ply"
        if [ -d "$ply_dir" ] && [ -n "$(ls "$ply_dir"/iter_*.ply 2>/dev/null)" ]; then
            do_resume=true
        else
            echo "No checkpoint in $ply_dir, starting from scratch for $scene."
        fi
    fi
    echo "Running train for $scene (base: $base_scene)${do_resume:+ [resume]}..."
    local train_cmd="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py \
        --dataset_type $DATASET_TYPE \
        --dataset_name $scene \
        --dataset_path $DATASET_ROOT/$base_scene \
        --selfocc_offset_max $offset \
        --geometry_threshold $geometry_threshold"
    if [ "$do_resume" = true ]; then
        train_cmd="$train_cmd --resume"
    else
        train_cmd="$train_cmd --ply_path $DATASET_ROOT/3dgrt/${base_scene}_refined.ply"
    fi
    eval $train_cmd
}

# Function to run render for a scene
run_render() {
    local scene=$1
    # Extract base scene name (e.g., "lego" from "lego2")
    local base_scene=$(echo $scene | sed 's/[0-9]*$//')
    # Use base scene parameters if available, otherwise use defaults
    local offset=${SCENE_PARAMS[$base_scene,offset]:-"0.5"}
    local geometry_threshold=${SCENE_PARAMS[$base_scene,geometry_threshold]:-"0.3"}
    local ply_dir="$OUTPUT_ROOT/$scene/ply"
    local envmap_dir="$OUTPUT_ROOT/$scene/envmap"
    local expected_ply="$ply_dir/iter_${ITERATION}.ply"
    local ply_path="$expected_ply"
    local iter_padded
    # If expected iter PLY missing (e.g. training was killed), use latest checkpoint
    if [ ! -f "$expected_ply" ]; then
        local latest_ply
        latest_ply=$(ls "$ply_dir"/iter_*.ply 2>/dev/null | sort -t_ -k2 -n | tail -n1)
        if [ -n "$latest_ply" ]; then
            ply_path="$latest_ply"
            local iter_num
            iter_num=$(echo "$latest_ply" | sed -n 's/.*iter_\([0-9]*\)\.ply/\1/p')
            iter_padded=$(printf "%04d" "$iter_num")
            echo "Note: iter_${ITERATION}.ply not found; using latest checkpoint $ply_path for render."
        else
            iter_padded=$(printf "%04d" $ITERATION)
        fi
    else
        iter_padded=$(printf "%04d" $ITERATION)
    fi
    local envmap_path="$envmap_dir/optimized_sgs_${iter_padded}.npy"
    if [ ! -f "$envmap_path" ]; then
        local latest_envmap
        latest_envmap=$(ls "$envmap_dir"/optimized_sgs_*.npy 2>/dev/null | sort -t_ -k2 -n | tail -n1)
        if [ -n "$latest_envmap" ]; then
            envmap_path="$latest_envmap"
            echo "Note: using latest envmap $envmap_path"
        fi
    fi
    echo "Running render for $scene (ply: $ply_path)..."
    
    # Build render command with optional relight flag
    local render_cmd="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python render.py \
        --dataset_type $DATASET_TYPE \
        --dataset_name $scene \
        --dataset_path $DATASET_ROOT/$base_scene \
        --ply_path $ply_path \
        --render_spp 64 \
        --envmap_init_path $envmap_path"
    
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
# Separate scene names from option arguments
scene_names=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda_device)
            CUDA_DEVICE="$2"
            echo "Using CUDA device: $CUDA_DEVICE"
            shift 2
            ;;
        --iteration)
            ITERATION="$2"
            echo "Using iteration: $ITERATION"
            shift 2
            ;;
        --resume)
            RESUME=true
            echo "Resume: will load latest checkpoint from each scene's output dir"
            shift
            ;;
        *)
            # Check if it's a number (old iteration syntax support)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                ITERATION="$1"
                echo "Using iteration: $ITERATION"
            else
                # Add to scene names list
                scene_names+=($1)
            fi
            shift
            ;;
    esac
done

# Main execution
if [ ${#scene_names[@]} -eq 0 ]; then
    # Run all scenes if no scene names provided
    echo "Running all scenes with CUDA device $CUDA_DEVICE and iteration $ITERATION..."
    for scene in lego hotdog armadillo ficus; do
        run_all $scene
    done
else
    # Run only specified scenes
    echo "Running specified scenes: ${scene_names[@]} with CUDA device $CUDA_DEVICE and iteration $ITERATION..."
    for scene in "${scene_names[@]}"; do
        run_all $scene
    done
fi

echo "Done!"
