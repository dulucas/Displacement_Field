import os.path as osp
import sys
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.nn.functional as F

from config import config
from dataloader import get_train_loader
from df import Displacement_Field
from datasets.nyu import NYUDataset

from utils.init_func import init_weight
from misc.utils import get_params
from engine.lr_policy import PolyLR
from engine.engine import Engine

class Mseloss(MSELoss):
    def __init__(self):
        super(Mseloss, self).__init__()

    def forward(self, input, target, mask=None):
        if mask is not None:
            input = input.squeeze(1)
            input = torch.mul(input, mask)
            target = torch.mul(target, mask)
            loss = F.mse_loss(input, target, reduction=self.reduction)

            return loss

parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    seed = config.seed
    torch.manual_seed(seed)

    train_loader, train_sampler = get_train_loader(engine, NYUDataset)

    criterion = Mseloss()
    BatchNorm2d = nn.BatchNorm2d

    model = Displacement_Field()
    init_weight(model.displacement_net, nn.init.xavier_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum)
    base_lr = config.lr

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if engine.continue_state_object:
        engine.restore_checkpoint()
    model.zero_grad()
    model.train()

    optimizer = torch.optim.Adam(params=get_params(model),
                                 lr=base_lr)
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)

    for epoch in range(engine.state.epoch, config.nepochs):
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)
            minibatch = dataloader.next()

            imgs = minibatch['guidance']
            deps = minibatch['data']
            gts = minibatch['label']
            masks = minibatch['mask']

            imgs = imgs.cuda()
            imgs = torch.autograd.Variable(imgs)
            deps = deps.cuda()
            deps = torch.autograd.Variable(deps)
            gts = gts.cuda()
            gts = torch.autograd.Variable(gts)
            masks = masks.cuda()
            masks = torch.autograd.Variable(masks)

            pred = model(imgs, deps)
            loss = criterion(pred, gts, masks)
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.6f' % float(loss)

            pbar.set_description(print_str, refresh=False)

        if (epoch == (config.nepochs - 1)) or (epoch % config.snapshot_iter == 0):
            engine.save_and_link_checkpoint(config.snapshot_dir,
                                            config.log_dir,
                                            config.log_dir_link)
