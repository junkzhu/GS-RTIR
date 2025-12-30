通过网盘分享的文件：TensoIR.zip
链接: https://pan.baidu.com/s/1sxG5vFs24gzOFAoLR-fdQQ?pwd=vapz 提取码: vapz

单个armadillo数据集： （建议使用，小一点..）
链接: https://pan.baidu.com/s/1cAjMdc3vL0H6yju9VHHueA?pwd=va9j 提取码: va9j 


使用方法..(ai写的)

# Usage

## Training on TensoIR Dataset
```bash
# Train a single scene
./train_TensoIR.sh lego

# Train a single scene and specify which iteration's ply file to use for render/metrics
./train_TensoIR.sh 200 lego

# Train multiple scenes
./train_TensoIR.sh lego hotdog ficus

# Run a debug instance (automatically creates lego2 folder)
./train_TensoIR.sh lego2
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