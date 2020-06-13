import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 304

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.label_dir = C.abs_dir.split(osp.sep)[-2]
C.root_dir = C.abs_dir[:C.abs_dir.index('model')]
C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.label_dir, C.this_dir))
C.log_dir_link = osp.join(C.abs_dir, 'log')
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

"""Data Dir and Weight Dir"""
C.data_source = '/home/duy/phd/Displacement_Field/dataset/nyu_depth_v2_labeled.mat'
C.train_test_splits = '/home/duy/phd/Displacement_Field/dataset/nyuv2_splits.mat'
C.is_test = False

"""Path Config"""

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir, 'lib'))

"""Image Config"""
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
C.use_gauss_blur = True
C.gaussian_kernel_range = [.3, 1]
C.max_kernel = 51
C.downsampling_scale = 8
C.interpolation = 'LINEAR'
C.use_updown_sampling = False
C.target_size = 320
C.image_height = 320
C.image_width = 320
C.num_train_imgs = 795
C.num_eval_imgs = 654

""" Settings for network, this would be different for each kind of model"""
C.fix_bias = False
C.fix_bn = False
C.bn_eps = 1e-5
C.bn_momentum = 0.1
C.loss_weight = None
C.pretrained_model = None

"""Train Config"""
C.lr = 1e-3
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-6
C.batch_size = 1
C.nepochs = 20
C.niters_per_epoch = 795
C.num_workers = 4
C.train_scale_array = [1., 1.5, 2, 2.5, 4]
C.business_lr_ratio = 1.0
C.aux_loss_ratio = 1

"""Display Config"""
C.snapshot_iter = 5
C.record_info_iter = 20
C.display_iter = 50

def open_tensorboard():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
