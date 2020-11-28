import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from layers.basic_module import ConvBnRelu, ConvBnLeakyRelu, SeparableConvBnLeakyRelu, SeparableConvBnRelu, \
        SELayer, ChannelAttention, BNRefine, RefineResidual, AttentionRefinement, GlobalAvgPool2d, \
        FeatureFusion

def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.cuda()

def addpositionalembed(tensor):
    n,c,h,w = tensor.size()
    pos_embed = positionalencoding2d(c,h,w)
    pos_embed = pos_embed.unsqueeze(0).repeat(n,1,1,1)
    tensor += pos_embed
    return tensor

def concatpositionalembed(tensor):
    n,c,h,w = tensor.size()
    pos_embed = positionalencoding2d(c,h,w)
    pos_embed = pos_embed.unsqueeze(0).repeat(n,1,1,1)
    tensor = torch.cat((tensor, pos_embed), 1)
    return tensor

def multipositionalembed(tensor):
    n,c,h,w = tensor.size()
    pos_embed = positionalencoding2d(c,h,w)
    pos_embed = pos_embed.unsqueeze(0).repeat(n,1,1,1)
    tensor *= pos_embed
    return tensor

###########################################################

class Unet(nn.Module):
    def __init__(self, n_channels, rgb_channels, n_classes):
        super(Unet, self).__init__()

        self.positionalembed = addpositionalembed

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

        self.depth_down_embed0 = ConvBnLeakyRelu(16, 8, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_embed1 = ConvBnLeakyRelu(32, 16, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_embed2 = ConvBnLeakyRelu(64, 32, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_embed3 = ConvBnLeakyRelu(128, 64, 3, 1, 1, 1, 1,\
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

        self.depth_up_embed0 = RefineResidual(128, 64, relu_layer='LeakyReLU', \
                                     has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_embed1 = RefineResidual(128, 64, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_embed2 = RefineResidual(64, 32, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_embed3 = RefineResidual(32, 16, relu_layer='LeakyReLU', \
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

        self.rgb_down_layer0 = ConvBnRelu(rgb_channels, 8, 3, 1, 1, 1, 1,\
                                     has_bn=True, \
                                     inplace=True, has_bias=True)
        self.rgb_down_layer1 = ConvBnRelu(8, 16, 3, 1, 1, 1, 1,\
                                     has_bn=True, \
                                     inplace=True, has_bias=True)
        self.rgb_down_layer2 = ConvBnRelu(16, 32, 3, 1, 1, 1, 1,\
                                     has_bn=True, \
                                     inplace=True, has_bias=True)
        self.rgb_down_layer3 = ConvBnRelu(32, 64, 3, 1, 1, 1, 1,\
                                     has_bn=True, \
                                     inplace=True, has_bias=True)

        self.rgb_down_embed0 = ConvBnRelu(16, 8, 3, 1, 1, 1, 1,\
                                     has_bn=True, \
                                     inplace=True, has_bias=True)
        self.rgb_down_embed1 = ConvBnRelu(32, 16, 3, 1, 1, 1, 1,\
                                     has_bn=True, \
                                     inplace=True, has_bias=True)
        self.rgb_down_embed2 = ConvBnRelu(64, 32, 3, 1, 1, 1, 1,\
                                     has_bn=True, \
                                     inplace=True, has_bias=True)
        self.rgb_down_embed3 = ConvBnRelu(128, 64, 3, 1, 1, 1, 1,\
                                     has_bn=True, \
                                     inplace=True, has_bias=True)

        self.rgb_refine_layer0 = RefineResidual(8, 8, relu_layer='ReLU', \
                                     has_bias=True, has_relu=True)
        self.rgb_refine_layer1 = RefineResidual(16, 16, relu_layer='ReLU', \
                                    has_bias=True, has_relu=True)
        self.rgb_refine_layer2 = RefineResidual(32, 32, relu_layer='ReLU', \
                                    has_bias=True, has_relu=True)
        self.rgb_refine_layer3 = RefineResidual(64, 64, relu_layer='ReLU', \
                                    has_bias=True, has_relu=True)

        self.output_layer = ConvBnRelu(4, 2, 3, 1, 1, 1, 1,\
                                     has_bn=False, \
                                     has_relu=False, inplace=True, has_bias=True)

    def forward(self, rgb, x):
        output = []
        #### RGB ####
        r1 = self.rgb_down_layer0(rgb)
        r1 = self.down_scale(r1)
        r1 = self.positionalembed(r1)
        #r1 = self.rgb_down_embed0(r1)

        r2 = self.rgb_down_layer1(r1)
        r2 = self.down_scale(r2)
        r2 = self.positionalembed(r2)
        #r2 = self.rgb_down_embed1(r2)

        r3 = self.rgb_down_layer2(r2)
        r3 = self.down_scale(r3)
        r3 = self.positionalembed(r3)
        #r3 = self.rgb_down_embed2(r3)

        r4 = self.rgb_down_layer3(r3)
        r4 = self.down_scale(r4)
        r4 = self.positionalembed(r4)
        #r4 = self.rgb_down_embed3(r4)

        r1 = self.rgb_refine_layer0(r1)
        r2 = self.rgb_refine_layer1(r2)
        r3 = self.rgb_refine_layer2(r3)
        r4 = self.rgb_refine_layer3(r4)
        #### Depth ####
        x1 = self.depth_down_layer0(x)
        x1 = self.down_scale(x1)
        #x1 = self.positionalembed(x1)
        #x1 = self.depth_down_embed0(x1)

        x1 = self.depth_down_layer1(x1)
        x1 = self.down_scale(x1)
        #x1 = self.positionalembed(x1)
        #x1 = self.depth_down_embed1(x1)

        x2 = self.depth_down_layer2(x1)
        x2 = self.down_scale(x2)
        #x2 = self.positionalembed(x2)
        #x2 = self.depth_down_embed2(x2)

        x = self.depth_down_layer3(x2)
        x = self.down_scale(x)
        #x = self.positionalembed(x)
        #x = self.depth_down_embed3(x)

        #x = self.depth_up_embed0(x)
        x = self.depth_up_layer0(x)
        #out = self.depth_out_layer0(x)
        x = x + r4
        x = self.up_scale(x)

        #x = self.positionalembed(x)
        #x = self.depth_up_embed1(x)
        x = self.depth_up_layer1(x)
        #out = self.depth_out_layer1(x)
        x = x + x2 + r3
        x = self.up_scale(x)

        #x = self.positionalembed(x)
        #x = self.depth_up_embed2(x)
        x = self.depth_up_layer2(x)
        #out = self.depth_out_layer2(x)
        x = x + x1 + r2
        x = self.up_scale(x)

        #x = self.positionalembed(x)
        #x = self.depth_up_embed3(x)
        x = self.depth_up_layer3(x)
        x = x + r1
        x = self.positionalembed(x)

        x = self.up_scale(x)
        x = self.refine_layer0(x)
        x = self.refine_layer1(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
   unet = Unet(1,1,1).cuda()
   img = torch.rand(1,1,320,320).cuda()
   depth = torch.rand(1,1,320,320).cuda()
   x = unet(img, depth)
   print(x.shape)
