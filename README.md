# Displacement_Field

## Installation

### Requirements:
- PyTorch >= 0.4
- OpenCV
- CUDA >= 8.0(Only tested with CUDA >= 8.0)

### Data Preparation
```bash
sh download.sh
```

### Training
```bash
#Use depth only as input
cd model/nyu/df_nyu_depth_only
python train.py -d 0

#Use RGB image as guidance
cd model/nyu/df_nyu_rgb_guidance
python train.py -d 0
```
