## 数据集
数据集链接：
链接: https://pan.baidu.com/s/1KGnlYt-Bfgt_x9mPz4kQAw?pwd=puau 提取码: puau 
如果已有包含先验的数据集，可以直接下链接里面的_3dgrt模型文件。_3dgrt模型文件和数据集里的3dgrt文件一致，不需要重新粘贴。

## 说明
常用constant中的参数：
--stride 合成数据集都是很平滑的视角变化，跳过几个不影响指标，可以快速看render结果
--envmap_optimization 关掉会用GT envmap 即sunset
TRAIN_UPSAMPLE_ITER 会默认进行训练照片分辨率调整，如[200,400], 1-200分辨率 800/4 = 200, 200-400分辨率 800/2 = 400, 400-800分辨率 800/1 = 800

训练轮次和学习率在 train.yaml

目前结果记录：
https://docs.qq.com/sheet/DR3BYTWJWd1N4ZHFF?tab=BB08J2
理论上就是如下命令能跑出来的结果：
./train_TensoIR.sh lego
./train_Synthetic4Relight.sh jugs
...


使用方法..(ai写的，但是是对的)

# Usage

## Training on TensoIR Dataset
```bash
# Train a single scene
./train_TensoIR.sh lego

# Train a single scene and specify which iteration's ply file to use for render/metrics
./train_TensoIR.sh lego --iteration 200 #这个是控制relight用哪个ply的，和训练次数无关

# Train multiple scenes
./train_TensoIR.sh lego hotdog ficus

# Run a debug instance (automatically creates lego2 folder)
./train_TensoIR.sh lego2 --cuda_device 1 #便于debug
```

## Training on Synthetic4Relight Dataset
```bash
# Train a single scene
./train_Synthetic4Relight.sh jugs

# Train a single scene and specify which iteration's ply file to use for render/metrics
./train_Synthetic4Relight.sh 200 jugs

# Train multiple scenes
./train_Synthetic4Relight.sh jugs chair hotdog

# Run a debug instance (automatically creates jugs2 folder)
./train_Synthetic4Relight.sh jugs2
```

## Step Switches
Each training script has step switches at the top to control which steps to execute:

```bash
# Enable/disable switches for each step
enable_refine=true
enable_train=true
enable_render=true
enable_metrics=true
enable_relight=true  # Switch for relight functionality
```

- `enable_refine`: Whether to perform model refinement step
- `enable_train`: Whether to perform training step
- `enable_render`: Whether to perform rendering step
- `enable_metrics`: Whether to compute evaluation metrics
- `enable_relight`: Whether to enable relight functionality

## CUDA Device Selection
You can specify which CUDA device to use with the `--cuda_device` parameter:

```bash
# Train on CUDA device 0
./train_TensoIR.sh lego --cuda_device 0

# Train on CUDA device 3 with a specific iteration
./train_TensoIR.sh hotdog --cuda_device 3 --iteration 100

# Same syntax works for Synthetic4Relight dataset
./train_Synthetic4Relight.sh jugs --cuda_device 1
```

The parameter works regardless of its position in the command line. If not specified, the default CUDA device (0) will be used.

## Multiple Instances
You can run multiple debug instances simultaneously, each with its own output folder and CUDA device:

```bash
# Run two instances on different CUDA devices
./train_TensoIR.sh lego2 --cuda_device 0
./train_TensoIR.sh lego3 --cuda_device 1
```

The system automatically identifies the base scene name (e.g., lego) and uses the numeric suffix (e.g., 2, 3) to create unique output directories.