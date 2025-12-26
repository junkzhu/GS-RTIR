#!/bin/bash

#refine
CUDA_VISIBLE_DEVICES=0 python refine/refine_Synthetic4Relight.py --dataset_type Synthetic4Relight --dataset_name jugs --dataset_path /home/zjk/datasets/Synthetic4Relight/jugs --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/jugs.pt --refine_path /home/zjk/datasets/Synthetic4Relight/3dgrt/jugs_refined.ply
CUDA_VISIBLE_DEVICES=0 python refine/refine_Synthetic4Relight.py --dataset_type Synthetic4Relight --dataset_name chair --dataset_path /home/zjk/datasets/Synthetic4Relight/chair --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/chair.pt --refine_path /home/zjk/datasets/Synthetic4Relight/3dgrt/chair_refined.ply
CUDA_VISIBLE_DEVICES=0 python refine/refine_Synthetic4Relight.py --dataset_type Synthetic4Relight --dataset_name air_baloons --dataset_path /home/zjk/datasets/Synthetic4Relight/air_baloons --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/air_baloons.pt --refine_path /home/zjk/datasets/Synthetic4Relight/3dgrt/air_baloons_refined.ply
CUDA_VISIBLE_DEVICES=0 python refine/refine_Synthetic4Relight.py --dataset_type Synthetic4Relight --dataset_name hotdog --dataset_path /home/zjk/datasets/Synthetic4Relight/hotdog --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/hotdog.pt --refine_path /home/zjk/datasets/Synthetic4Relight/3dgrt/hotdog_refined.ply

#train
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type Synthetic4Relight --dataset_name jugs --dataset_path /home/zjk/datasets/Synthetic4Relight/jugs --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/jugs_refined.ply --selfocc_offset_max 0.1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type Synthetic4Relight --dataset_name chair --dataset_path /home/zjk/datasets/Synthetic4Relight/chair --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/chair_refined.ply --selfocc_offset_max 0.1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type Synthetic4Relight --dataset_name air_baloons --dataset_path /home/zjk/datasets/Synthetic4Relight/air_baloons --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/air_baloons_refined.ply --selfocc_offset_max 0.3
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type Synthetic4Relight --dataset_name hotdog --dataset_path /home/zjk/datasets/Synthetic4Relight/hotdog --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/hotdog_refined.ply --selfocc_offset_max 0.1

#render & metric
CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type Synthetic4Relight --dataset_name jugs --dataset_path /home/zjk/datasets/Synthetic4Relight/jugs --ply_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/jugs/ply/iter_299.ply --render_spp 64 --envmap_init_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/jugs/envmap/optimized_sgs_0299.npy --relight --envmap_root /home/zjk/datasets/Synthetic4Relight/Environment_Maps
CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type Synthetic4Relight --dataset_name jugs --dataset_path /home/zjk/datasets/Synthetic4Relight/jugs --envmap_root /home/zjk/datasets/Synthetic4Relight/Environment_Maps --relight 

CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type Synthetic4Relight --dataset_name chair --dataset_path /home/zjk/datasets/Synthetic4Relight/chair --ply_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/chair/ply/iter_299.ply --render_spp 64 --envmap_init_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/chair/envmap/optimized_sgs_0299.npy --relight --envmap_root /home/zjk/datasets/Synthetic4Relight/Environment_Maps
CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type Synthetic4Relight --dataset_name chair --dataset_path /home/zjk/datasets/Synthetic4Relight/chair --envmap_root /home/zjk/datasets/Synthetic4Relight/Environment_Maps --relight 

CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type Synthetic4Relight --dataset_name air_baloons --dataset_path /home/zjk/datasets/Synthetic4Relight/air_baloons --ply_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/air_baloons/ply/iter_299.ply --render_spp 64 --envmap_init_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/air_baloons/envmap/optimized_sgs_0299.npy --relight --envmap_root /home/zjk/datasets/Synthetic4Relight/Environment_Maps
CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type Synthetic4Relight --dataset_name air_baloons --dataset_path /home/zjk/datasets/Synthetic4Relight/air_baloons --envmap_root /home/zjk/datasets/Synthetic4Relight/Environment_Maps --relight 

CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type Synthetic4Relight --dataset_name hotdog --dataset_path /home/zjk/datasets/Synthetic4Relight/hotdog --ply_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/hotdog/ply/iter_299.ply --render_spp 64 --envmap_init_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/hotdog/envmap/optimized_sgs_0299.npy --relight --envmap_root /home/zjk/datasets/Synthetic4Relight/Environment_Maps
CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type Synthetic4Relight --dataset_name hotdog --dataset_path /home/zjk/datasets/Synthetic4Relight/hotdog --envmap_root /home/zjk/datasets/Synthetic4Relight/Environment_Maps --relight 

echo "Done!"