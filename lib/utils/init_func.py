#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, business_layer=False,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
            if business_layer:
                nn.init.constant_(m.bias, 1)
            else:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, norm_layer):
            if m.weight is not None:
                m.eps = bn_eps
                m.momentum = bn_momentum
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            #group_decay.append(m.weight)
            #group_decay.append(m.bias)
            for p in m.parameters():
                yield p
        elif isinstance(m, nn.Conv2d):
            #group_decay.append(m.weight)
            #group_decay.append(m.bias)
            for p in m.parameters():
               yield p
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            #if m.weight is not None:
            #    group_decay.append(m.weight)
            #if m.bias is not None:
            #    group_decay.append(m.bias)
            for p in m.parameters():
               yield p
        else:
            #if 'weight' in dir(m) and m.weight is not None:
            #    group_decay.append(m.weight)
            #if 'bias' in dir(m) and m.bias is not None:
            #    group_decay.append(m.bias)
            for p in m.parameters():
                yield p

    #assert len(list(module.parameters())) == len(group_decay) + len(
    #    group_no_decay)
    #weight_group.append(dict(params=group_decay, lr=lr))
    #weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    #return weight_group
