import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from layers.basic_module import ConvBnRelu, ConvBnLeakyRelu, SeparableConvBnLeakyRelu, SeparableConvBnRelu, \
        SELayer, ChannelAttention, BNRefine, RefineResidual, AttentionRefinement, GlobalAvgPool2d, \
        FeatureFusion

###########################################################

class Unet(nn.Module):
    def __init__(self, n_channels, rgb_channels, n_classes):
        super(Unet, self).__init__()

        self.down_scale = nn.MaxPool2d(2)
        self.up_scale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.depth_down_layer0 = ConvBnLeakyRelu(n_channels, 8, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer1 = ConvBnLeakyRelu(8, 16, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer2 = ConvBnLeakyRelu(16, 32, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer3 = ConvBnLeakyRelu(32, 64, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)

        self.depth_up_layer0 = RefineResidual(64, 64, relu_layer='LeakyReLU', \
                                     has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_layer1 = RefineResidual(64, 32, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_layer2 = RefineResidual(32, 16, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_layer3 = RefineResidual(16, 8, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)

        self.depth_out_layer0 = RefineResidual(64, 1, relu_layer='LeakyReLU', \
                                     has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_out_layer1 = RefineResidual(32, 1, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_out_layer2 = RefineResidual(16, 1, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_out_layer3 = RefineResidual(8, 1, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)

        self.refine_layer0 = ConvBnLeakyRelu(8, 4, 3, 1, 1, 1, 1,\
                                    has_bn=True, leaky_alpha=0.3, \
                                    has_leaky_relu=True, inplace=True, has_bias=True)
        self.refine_layer1 = ConvBnLeakyRelu(4, 4, 3, 1, 1, 1, 1,\
                                    has_bn=True, leaky_alpha=0.3, \
                                    has_leaky_relu=True, inplace=True, has_bias=True)

        self.output_layer = ConvBnRelu(4, 2, 3, 1, 1, 1, 1,\
                                     has_bn=False, \
                                     has_relu=False, inplace=True, has_bias=True)

    def forward(self, x):
        #### Depth ####
        x1 = self.depth_down_layer0(x)
        x1 = self.down_scale(x1)
        x1 = self.depth_down_layer1(x1)
        x1 = self.down_scale(x1)
        x2 = self.depth_down_layer2(x1)
        x2 = self.down_scale(x2)
        x = self.depth_down_layer3(x2)
        x = self.down_scale(x)

        x = self.depth_up_layer0(x)
        x = self.up_scale(x)
        x = self.depth_up_layer1(x)
        x = x + x2
        x = self.up_scale(x)
        x = self.depth_up_layer2(x)
        x = x + x1
        x = self.up_scale(x)
        x = self.depth_up_layer3(x)
        x = self.up_scale(x)
        x = self.refine_layer0(x)
        x = self.refine_layer1(x)
        x = self.output_layer(x)
        return x
