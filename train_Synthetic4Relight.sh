#!/bin/bash

#refine
CUDA_VISIBLE_DEVICES=0 python refine/refine_Synthetic4Relight.py --dataset_type Synthetic4Relight --dataset_name jugs --dataset_path /home/zjk/datasets/Synthetic4Relight/jugs --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/jugs.pt --refine_path /home/zjk/datasets/Synthetic4Relight/3dgrt/jugs_refined.ply
CUDA_VISIBLE_DEVICES=0 python refine/refine_Synthetic4Relight.py --dataset_type Synthetic4Relight --dataset_name chair --dataset_path /home/zjk/datasets/Synthetic4Relight/chair --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/chair.pt --refine_path /home/zjk/datasets/Synthetic4Relight/3dgrt/chair_refined.ply
CUDA_VISIBLE_DEVICES=0 python refine/refine_Synthetic4Relight.py --dataset_type Synthetic4Relight --dataset_name air_baloons --dataset_path /home/zjk/datasets/Synthetic4Relight/air_baloons --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/air_baloons.pt --refine_path /home/zjk/datasets/Synthetic4Relight/3dgrt/air_baloons_refined.ply
CUDA_VISIBLE_DEVICES=0 python refine/refine_Synthetic4Relight.py --dataset_type Synthetic4Relight --dataset_name hotdog --dataset_path /home/zjk/datasets/Synthetic4Relight/hotdog --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/hotdog.pt --refine_path /home/zjk/datasets/Synthetic4Relight/3dgrt/hotdog_refined.ply

#train
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type Synthetic4Relight --dataset_name jugs --dataset_path /home/zjk/datasets/Synthetic4Relight/jugs --ply_path /home/zjk/datasets/Synthetic4Relight/3dgrt/jugs_refined.ply --selfocc_offset_max 0.1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type Synthetic4Relight --dataset_name chair --dataset_path /home/zjk/datasets/Synthetic4Relight/chair --ply_path /home/zjk/datasets/Synthetic4Relight/chair.ply --selfocc_offset_max 0.1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type Synthetic4Relight --dataset_name air_baloons --dataset_path /home/zjk/datasets/Synthetic4Relight/air_baloons --ply_path /home/zjk/datasets/Synthetic4Relight/air_baloons.ply --selfocc_offset_max 0.1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_type Synthetic4Relight --dataset_name hotdog --dataset_path /home/zjk/datasets/Synthetic4Relight/hotdog --ply_path /home/zjk/datasets/Synthetic4Relight/hotdog.ply --selfocc_offset_max 0.1

#render
CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type Synthetic4Relight --dataset_name jugs --dataset_path /home/zjk/datasets/Synthetic4Relight/jugs --ply_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/jugs/ply/iter_499.ply --render_spp 256 --envmap_init_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/jugs/envmap/optimized_sgs_0499.npy
CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type Synthetic4Relight --dataset_name chair --dataset_path /home/zjk/datasets/Synthetic4Relight/chair --ply_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/chair/ply/iter_499.ply --render_spp 256 --envmap_init_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/chair/envmap/optimized_sgs_0499.npy
CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type Synthetic4Relight --dataset_name air_baloons --dataset_path /home/zjk/datasets/Synthetic4Relight/air_baloons --ply_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/air_baloons/ply/iter_499.ply --render_spp 256 --envmap_init_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/air_baloons/envmap/optimized_sgs_0499.npy
CUDA_VISIBLE_DEVICES=0 python render.py --dataset_type Synthetic4Relight --dataset_name hotdog --dataset_path /home/zjk/datasets/Synthetic4Relight/hotdog --ply_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/hotdog/ply/iter_499.ply --render_spp 256 --envmap_init_path /home/zjk/code/GS-RTIR/outputs/Synthetic4Relight/hotdog/envmap/optimized_sgs_0499.npy

#metrics
CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type Synthetic4Relight --dataset_name jugs --dataset_path /home/zjk/datasets/Synthetic4Relight/jugs
CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type Synthetic4Relight --dataset_name chair --dataset_path /home/zjk/datasets/Synthetic4Relight/chair
CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type Synthetic4Relight --dataset_name air_baloons --dataset_path /home/zjk/datasets/Synthetic4Relight/air_baloons
CUDA_VISIBLE_DEVICES=0 python metrics.py --dataset_type Synthetic4Relight --dataset_name hotdog --dataset_path /home/zjk/datasets/Synthetic4Relight/hotdog

echo "Done!"