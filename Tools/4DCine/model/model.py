#################################################
# Nicolo Savioli, PhD - Imperial College London # 
#################################################

import math
from torch                 import nn
import torch
import torch.nn.functional as F
import cv2
import numpy               as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .unet_parts import *
from collections import OrderedDict
from torch.nn import init
import numpy as np

##########################
# ORIGINAL U-NET MODEL   #
##########################

# Implements UNET models: https://arxiv.org/abs/1505.04597

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.pooling      = pooling
        self.bn2          = nn.BatchNorm2d(self.out_channels)

        self.conv1        = conv3x3(self.in_channels, self.out_channels)
        self.conv2        = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='add', up_mode='upsample'):
        super(UpConv, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.merge_mode   = merge_mode
        self.up_mode      = up_mode
        self.bn2          = nn.BatchNorm2d(self.out_channels)

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNet_deep(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, combine, flag, num_classes=4, in_channels=100,out_channels=100, depth=3, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet_deep, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes  = num_classes
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.start_filts  = start_filts
        self.depth        = depth
        self.down_convs   = []
        self.up_convs_1   = []
        self.up_convs_2   = []
        self.combine      = combine
        self.flag         = flag

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks

        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs_1.append(up_conv)
            if self.combine:
                up_convc = UpConv(ins, outs, up_mode=up_mode,
                    merge_mode=merge_mode)
                self.up_convs_2.append(up_convc)

        if self.combine:
            self.conv_final_seg = conv1x1(outs, self.num_classes*self.out_channels)
            self.conv_final_img = conv1x1(outs, self.out_channels)
        else:
            if self.flag:
                self.conv_final = conv1x1(outs, self.num_classes* self.out_channels)
            else:
                self.conv_final = conv1x1(outs, self.out_channels)

        # Down - add the list of modules to current module
        self.down_convs          = nn.ModuleList(self.down_convs)
        
        # Up 
        self.up_convs_out_1      = nn.ModuleList(self.up_convs_1)
        if self.combine:
            self.up_convs_out_2  = nn.ModuleList(self.up_convs_2)


        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        encoder_outs = []
        img          = x
        y            = None
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
            if self.combine:
                y   = x

        for i, module in enumerate(self.up_convs_out_1):
            before_pool1 = encoder_outs[-(i+2)]
            x = module(before_pool1, x)
        
        if self.combine:
            for i, module in enumerate(self.up_convs_out_2):
                before_pool2 = encoder_outs[-(i+2)]
                y = module(before_pool2, y)
        

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        if self.combine:
            y = self.conv_final_seg(y).view(1, 4, img.size(1),\
                                img.size(2), img.size(3))
            x = self.conv_final_img(x)
 
        else:
            if self.flag:
                x = self.conv_final(x).view(1, 4, img.size(1),\
                                img.size(2), img.size(3))
            else:
                x = self.conv_final(x)
        if self.combine:
            return x,y
        else:
            return x


##################
### U-Net Img ####
##################

'''
class GeneratorUnetImg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(GeneratorUnetImg, self).__init__()
        print("\n\n\n ... U-NET RECONTRUCTION")
        self.model = UNet_deep(False,False)    
    def forward(self, x):
        return self.model(x)
'''

class GeneratorUnetImg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(GeneratorUnetImg, self).__init__()
        self.bilinear = bilinear
        self.inc        = DoubleConv(in_channels, 64)
        self.down1      = Down(64, 128)
        self.down2      = Down(128, 256)
        self.down3      = Down(256, 512)
        factor          = 2 if bilinear else 1
        self.down4      = Down(512, 1024 // factor)
        self.up_img_1   = Up(1024, 512 // factor, bilinear)
        self.up_img_2   = Up(512, 256 // factor, bilinear)
        self.up_img_3   = Up(256, 128 // factor, bilinear)
        self.up_img_4   = Up(128, 64, bilinear)
        self.outc_seg   = OutConv(64, out_channels)
        

    def forward(self, x):
        img      = x
        x1       = self.inc(x)
        x2       = self.down1(x1)
        x3       = self.down2(x2)
        x4       = self.down3(x3)
        x5       = self.down4(x4)
        x_img_1  = self.up_img_1(x5, x4)
        x_img_2  = self.up_img_2(x_img_1, x3)
        x_img_3  = self.up_img_3(x_img_2, x2)
        x_img_4  = self.up_img_4(x_img_3, x1)
        return self.outc_seg(x_img_4)



##################
### U-Net Seg ####
##################

'''
class GeneratorUnetSeg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(GeneratorUnetSeg, self).__init__()
        print("\n\n\n ... U-NET SEGMENTATION")
        self.model = UNet_deep(False,True)
    
    def forward(self, x):
        return self.model(x)
'''

class GeneratorUnetSeg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(GeneratorUnetSeg, self).__init__()
        self.bilinear = bilinear
        self.inc        = DoubleConv(in_channels, 64)
        self.down1      = Down(64, 128)
        self.down2      = Down(128, 256)
        self.down3      = Down(256, 512)
        factor          = 2 if bilinear else 1
        self.down4      = Down(512, 1024 // factor)
        self.up_seg_1   = Up(1024, 512 // factor, bilinear)
        self.up_seg_2   = Up(512, 256 // factor, bilinear)
        self.up_seg_3   = Up(256, 128 // factor, bilinear)
        self.up_seg_4   = Up(128, 64, bilinear)
        self.outc_seg   = OutConv(64, 4*out_channels)
        

    def forward(self, x):
        img      = x
        x1       = self.inc(x)
        x2       = self.down1(x1)
        x3       = self.down2(x2)
        x4       = self.down3(x3)
        x5       = self.down4(x4)
        x_seg_1  = self.up_seg_1(x5, x4)
        x_seg_2  = self.up_seg_2(x_seg_1, x3)
        x_seg_3  = self.up_seg_3(x_seg_2, x2)
        x_seg_4  = self.up_seg_4(x_seg_3, x1)
        logits   = self.outc_seg(x_seg_4).view(1, 4, img.size(1),\
                           img.size(2), img.size(3))
        return logits

########################
### U-Net Seg + Img ####
########################

'''
class GeneratorUnetSegImg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(GeneratorUnetSegImg, self).__init__()
        print("\n\n\n ... U-NET SEGMENTATION+RECONSTRUCTION")
        self.model = UNet_deep(True,True)
    def forward(self, x):
        img,seg = self.model(x)
        return img,seg
'''
##########################################################################################################

class GeneratorUnetSegImg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(GeneratorUnetSegImg, self).__init__()
        self.bilinear = bilinear

        self.inc        = DoubleConv(in_channels, 64)
        self.down1      = Down(64, 128)
        self.down2      = Down(128, 256)
        self.down3      = Down(256, 512)
        factor          = 2 if bilinear else 1
        self.down4      = Down(512, 1024 // factor)
        self.up_img_1   = Up(1024, 512 // factor, bilinear)
        self.up_img_2   = Up(512, 256 // factor, bilinear)
        self.up_img_3   = Up(256, 128 // factor, bilinear)
        self.up_img_4   = Up(128, 64, bilinear)
        self.outc_seg   = OutConv(64, 4*out_channels)
        self.up_seg_1   = Up(1024, 512 // factor, bilinear)
        self.up_seg_2   = Up(512, 256 // factor, bilinear)
        self.up_seg_3   = Up(256, 128 // factor, bilinear)
        self.up_seg_4   = Up(128, 64, bilinear)
        self.outc_img   = OutConv(64, out_channels)
        

    def forward(self, x):
        img      = x
        x1       = self.inc(x)
        x2       = self.down1(x1)
        x3       = self.down2(x2)
        x4       = self.down3(x3)
        x5       = self.down4(x4)
        x_seg_1  = self.up_seg_1(x5, x4)
        x_seg_2  = self.up_seg_2(x_seg_1, x3)
        x_seg_3  = self.up_seg_3(x_seg_2, x2)
        x_seg_4  = self.up_seg_4(x_seg_3, x1)
        #############################
        x_img_1  = self.up_img_1(x5, x4)
        x_img_2  = self.up_img_2(x_img_1, x3)
        x_img_3  = self.up_img_3(x_img_2, x2)
        x_img_4  = self.up_img_4(x_img_3, x1)
        logits   = self.outc_seg(x_seg_4).view(1, 4, img.size(1),\
                           img.size(2), img.size(3))
        img      = self.outc_img(x_img_4)
        return img,logits



class GeneratorUnetSegImgGAN(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(GeneratorUnetSegImgGAN, self).__init__()
        self.bilinear = bilinear

        self.inc        = DoubleConv(in_channels, 64)
        self.down1      = Down(64, 128)
        self.down2      = Down(128, 256)
        self.down3      = Down(256, 512)
        factor          = 2 if bilinear else 1
        self.down4      = Down(512, 1024 // factor)
        self.up_img_1   = Up(1024, 512 // factor, bilinear)
        self.up_img_2   = Up(512, 256 // factor, bilinear)
        self.up_img_3   = Up(256, 128 // factor, bilinear)
        self.up_img_4   = Up(128, 64, bilinear)
        self.outc_seg   = OutConv(64, 4*out_channels)
        self.up_seg_1   = Up(1024, 512 // factor, bilinear)
        self.up_seg_2   = Up(512, 256 // factor, bilinear)
        self.up_seg_3   = Up(256, 128 // factor, bilinear)
        self.up_seg_4   = Up(128, 64, bilinear)
        self.outc_img   = OutConv(64, out_channels)
        

    def forward(self, x, flag):
        img      = x
        x1       = self.inc(x)
        x2       = self.down1(x1)
        x3       = self.down2(x2)
        x4       = self.down3(x3)
        x5       = self.down4(x4)
        x5_noise = None
        if flag:
            noise    = torch.FloatTensor   (x5.size()).normal_(0, 1).cuda()
            x5_noise = torch.add           (x5,noise)
        else:
            x5_noise = x5
        x_seg_1  = self.up_seg_1(x5_noise,x4)
        x_seg_2  = self.up_seg_2(x_seg_1, x3)
        x_seg_3  = self.up_seg_3(x_seg_2, x2)
        x_seg_4  = self.up_seg_4(x_seg_3, x1)
        #############################
        x_img_1  = self.up_img_1(x5_noise,x4)
        x_img_2  = self.up_img_2(x_img_1, x3)
        x_img_3  = self.up_img_3(x_img_2, x2)
        x_img_4  = self.up_img_4(x_img_3, x1)
        logits   = self.outc_seg(x_seg_4).view(1, 4, img.size(1),\
                                               img.size(2), img.size(3))
        img      = self.outc_img(x_img_4)
        return img,logits


############
# SRGAN    #
############

# Implements SRGAN models: https://arxiv.org/abs/1609.04802


def swish(x):
    return x * torch.sigmoid(x)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input, pad_size):
        x = self.model(x)
        x = torch.cat((x, F.pad(skip_input, (pad_size,0,0,pad_size), 'constant', 0)), 1)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))


""" 
  Implements SRGAN models: https://arxiv.org/abs/1609.04802
"""

class Generator_resnet(nn.Module):
    def __init__(self, n_residual_blocks,in_channels,out_channels):
        super(Generator_resnet, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor   = 1

        self.conv1 = nn.Conv2d(in_channels, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        for i in range(int(self.upsample_factor/2)):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, out_channels, 9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(self.upsample_factor/2)):
            x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3(x)


##########################
### ResNet multi task ####
##########################
 
""" 
  Implements SRGAN models: https://arxiv.org/abs/1609.04802
"""

class Generator_resnet_mtk(nn.Module):
    def __init__(self, n_residual_blocks,in_channels,out_channels):
        super(Generator_resnet_mtk, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor   = 1
        self.dropout           = nn.Dropout(p=0.2)

        self.conv1 = nn.Conv2d(in_channels, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        for i in range(int(self.upsample_factor/2)):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.final_img  = nn.Conv2d(64, out_channels, 9, stride=1, padding=4)
        self.final_seg  = nn.Conv2d(64, 4*out_channels, 9, stride=1, padding=4)

    def forward(self, x):
        inputx = x
        x      = swish(self.conv1(x))
        y      = x.clone()

        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.dropout(self.bn2(self.conv2(y))) + x

        for i in range(int(self.upsample_factor/2)):
            x = self.__getattr__('upsample' + str(i+1))(x)
        
        img = self.final_img(x)
        seg = self.final_seg(x)        
        seg = self.final_seg(x).view(1, 4, inputx.size(1), inputx.size(2), inputx.size(3))
        return img,seg

##########################################################################################################

##########################
### Discriminator CNN ####
##########################

class Discriminator(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, out_channels, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return   torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

class DiscriminatorSeg(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DiscriminatorSeg, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, out_channels, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return   torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


class DiscriminatorImg(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DiscriminatorImg, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, out_channels, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return   torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)