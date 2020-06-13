# Displacement_Field
Official implementation of CVPR2020 paper **Predicting Sharp and Accurate Occlusion Boundaries in Monocular Depth Estimation Using Displacement Fields** [paper link](https://arxiv.org/abs/2002.12730)

NYUv2OC++ dataset [download link](https://drive.google.com/file/d/1Fk8uuH3oJJhyCN-4ffD3mdtCq2l4geJc/view)

## Visualization
### 1D example
![1D](./figure/toy.png)
------
### 2D example on blurry depth image(prediction of depth estimator)
![2D](./figure/displacement_field.png)
------
## Requirements:
- PyTorch >= 0.4
- OpenCV
- CUDA >= 8.0(Only tested with CUDA >= 8.0)

## Data Preparation
```bash
sh download.sh
```

## Training
```bash
#Use depth only as input
cd model/nyu/df_nyu_depth_only
python train.py -d 0

#Use RGB image as guidance
cd model/nyu/df_nyu_rgb_guidance
python train.py -d 0
```
## Citation
```bash
@InProceedings{Ramamonjisoa_2020_CVPR,
author = {Ramamonjisoa, Michael and Du, Yuming and Lepetit, Vincent},
title = {Predicting Sharp and Accurate Occlusion Boundaries in Monocular Depth Estimation Using Displacement Fields},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Acknowledgement
The code is based on [TorchSeg](https://github.com/ycszen/TorchSeg)
The NYUv2-OC++ is annotated manually by 4 Phd students major in computer vision. Special thanks to [Yang Xiao](https://youngxiao13.github.io/) and [Xuchong Qiu](https://imagine-lab.enpc.fr/staff-members/xuchong-qiu/) for their help in annotating the NYUv2-OC++ dataset.
