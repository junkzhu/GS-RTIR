通过网盘分享的文件：TensoIR.zip
链接: https://pan.baidu.com/s/1sxG5vFs24gzOFAoLR-fdQQ?pwd=vapz 提取码: vapz

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

## Multiple Instances
You can run multiple debug instances simultaneously, each with its own output folder:

```bash
# Run two instances of lego training simultaneously
./train_TensoIR.sh lego2
./train_TensoIR.sh lego3
```

The system automatically identifies the base scene name (e.g., lego) and uses the numeric suffix (e.g., 2, 3) to create unique output directories.