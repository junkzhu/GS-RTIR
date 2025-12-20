#!/bin/bash

#refine
CUDA_VISIBLE_DEVICES=1 python refine/refine_TensoIR.py --dataset_type TensoIR --dataset_name lego --dataset_path /home/zjk/datasets/TensoIR/lego --ply_path /home/zjk/datasets/TensoIR/3dgrt/lego.pt --refine_path /home/zjk/datasets/TensoIR/3dgrt/lego_refined.ply
CUDA_VISIBLE_DEVICES=1 python refine/refine_TensoIR.py --dataset_type TensoIR --dataset_name hotdog --dataset_path /home/zjk/datasets/TensoIR/hotdog --ply_path /home/zjk/datasets/TensoIR/3dgrt/hotdog.pt --refine_path /home/zjk/datasets/TensoIR/3dgrt/hotdog_refined.ply
CUDA_VISIBLE_DEVICES=1 python refine/refine_TensoIR.py --dataset_type TensoIR --dataset_name armadillo --dataset_path /home/zjk/datasets/TensoIR/armadillo --ply_path /home/zjk/datasets/TensoIR/3dgrt/armadillo.pt --refine_path /home/zjk/datasets/TensoIR/3dgrt/armadillo_refined.ply
CUDA_VISIBLE_DEVICES=1 python refine/refine_TensoIR.py --dataset_type TensoIR --dataset_name ficus --dataset_path /home/zjk/datasets/TensoIR/ficus --ply_path /home/zjk/datasets/TensoIR/3dgrt/ficus.pt --refine_path /home/zjk/datasets/TensoIR/3dgrt/ficus_refined.ply

#train
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_type TensoIR --dataset_name lego --dataset_path /home/zjk/datasets/TensoIR/lego --ply_path /home/zjk/datasets/TensoIR/3dgrt/lego_refined.ply --selfocc_offset_max 0.1
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_type TensoIR --dataset_name hotdog --dataset_path /home/zjk/datasets/TensoIR/hotdog --ply_path /home/zjk/datasets/TensoIR/3dgrt/hotdog_refined.ply --selfocc_offset_max 0.1
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_type TensoIR --dataset_name armadillo --dataset_path /home/zjk/datasets/TensoIR/armadillo --ply_path /home/zjk/datasets/TensoIR/3dgrt/armadillo_refined.ply --selfocc_offset_max 0.1
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_type TensoIR --dataset_name ficus --dataset_path /home/zjk/datasets/TensoIR/ficus --ply_path /home/zjk/datasets/TensoIR/3dgrt/ficus_refined.ply --selfocc_offset_max 0.1 --geometry_threshold 0.25

#render
CUDA_VISIBLE_DEVICES=1 python render.py --dataset_type TensoIR --dataset_name lego --dataset_path /home/zjk/datasets/TensoIR/lego --ply_path /home/zjk/code/GS-RTIR/outputs/TensoIR/lego/ply/iter_499.ply  --render_spp 256 --relight --envmap_init_path /home/zjk/code/GS-RTIR/outputs/TensoIR/lego/envmap/optimized_sgs_0499.npy
CUDA_VISIBLE_DEVICES=1 python render.py --dataset_type TensoIR --dataset_name hotdog --dataset_path /home/zjk/datasets/TensoIR/hotdog --ply_path /home/zjk/code/GS-RTIR/outputs/TensoIR/hotdog/ply/iter_499.ply --render_spp 256 --relight --envmap_init_path /home/zjk/code/GS-RTIR/outputs/TensoIR/hotdog/envmap/optimized_sgs_0499.npy
CUDA_VISIBLE_DEVICES=1 python render.py --dataset_type TensoIR --dataset_name armadillo --dataset_path /home/zjk/datasets/TensoIR/armadillo --ply_path /home/zjk/code/GS-RTIR/outputs/TensoIR/armadillo/ply/iter_499.ply  --render_spp 256 --relight --envmap_init_path /home/zjk/code/GS-RTIR/outputs/TensoIR/armadillo/envmap/optimized_sgs_0499.npy
CUDA_VISIBLE_DEVICES=1 python render.py --dataset_type TensoIR --dataset_name ficus --dataset_path /home/zjk/datasets/TensoIR/ficus --ply_path /home/zjk/code/GS-RTIR/outputs/TensoIR/ficus/ply/iter_499.ply  --render_spp 256 --relight --envmap_init_path /home/zjk/code/GS-RTIR/outputs/TensoIR/ficus/envmap/optimized_sgs_0499.npy

#metric
CUDA_VISIBLE_DEVICES=1 python metrics.py --dataset_type TensoIR --dataset_name lego --dataset_path /home/zjk/datasets/TensoIR/lego
CUDA_VISIBLE_DEVICES=1 python metrics.py --dataset_type TensoIR --dataset_name hotdog --dataset_path /home/zjk/datasets/TensoIR/hotdog
CUDA_VISIBLE_DEVICES=1 python metrics.py --dataset_type TensoIR --dataset_name armadillo --dataset_path /home/zjk/datasets/TensoIR/armadillo
CUDA_VISIBLE_DEVICES=1 python metrics.py --dataset_type TensoIR --dataset_name ficus --dataset_path /home/zjk/datasets/TensoIR/ficus
 
echo "Done!"