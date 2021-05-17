import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *
from torch.autograd import Variable
import numpy as np
import random


'''
Our model
'''

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
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
        target   = self.outc_img(x_img_4)
        source   = self.outc_img(x_img_4)
        return target,source


class ReplayBuffer:
    def __init__(self, max_size=10):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class DiscriminatorDomain1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DiscriminatorDomain1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3,   stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3,  stride=1, padding=1)
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
        self.relu = nn.ReLU(inplace=True)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, out_channels, 1, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return   torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)



class DiscriminatorDomain2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DiscriminatorDomain2, self).__init__()
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
        self.relu = nn.ReLU(inplace=True)


        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, out_channels, 1, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return   torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


class DiscriminatorDomain3(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DiscriminatorDomain3, self).__init__()
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
        self.relu = nn.ReLU(inplace=True)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, out_channels, 1, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return   torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


###################################################################
'''
Multimodal Unsupervised Image-to-Image Translation
https://arxiv.org/abs/1804.04732
'''
###################################################################



class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class MLP_MUNIT(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP_MUNIT, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class ContentEncoderMUNIT(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(ContentEncoderMUNIT, self).__init__()
        self.bilinear = bilinear
        self.inc      = DoubleConv(in_channels, 64)
        self.down1    = Down(64, 128)
        self.down2    = Down(128, 256)
        self.down3    = Down(256, 512)
        self.factor   = 2 if bilinear else 1
        self.down4    = Down(512, 1024 // self.factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, [x4,x3,x2,x1]

class StyleEncoderMUNIT(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(StyleEncoderMUNIT, self).__init__()
        self.bilinear = bilinear
        self.inc      = DoubleConv(in_channels, 64)
        self.down1    = Down(64, 128)
        self.down2    = Down(128, 256)
        self.down3    = Down(256, 512)
        self.factor   = 2 if bilinear else 1
        self.down4    = Down(512, 1024 // self.factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5,[x4,x3,x2,x1]


class StyleEncoderMUNIT(nn.Module):
    def __init__(self, in_channels, style_dim, dim=64, n_downsample=2):
        super(StyleEncoderMUNIT, self).__init__()

        # Initial conv block
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.ReLU(inplace=True)]

        # Downsampling
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
            dim *= 2

        # Downsampling with constant depth
        for _ in range(n_downsample - 2):
            layers += [nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.ReLU(inplace=True)]

        # Average pool and output layer
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class EncoderMUNIT(nn.Module):
    def __init__(self, in_channels,style_dim):
        super(EncoderMUNIT, self).__init__()
        self.content_encoder = ContentEncoderMUNIT(in_channels)
        self.style_encoder = StyleEncoderMUNIT(in_channels,style_dim)

    def forward(self, x):
        content_code,cf = self.content_encoder(x)
        style_code = self.style_encoder(x)
        return [content_code,cf], style_code

#################################
#            Decoder
#################################

class DoubleConvAdaIN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            AdaptiveInstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            AdaptiveInstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpAdaIN(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvAdaIN(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvAdaIN(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DecoderMUNIT(nn.Module):
    def __init__(self, out_channels, style_dim, bilinear=True):
        super(DecoderMUNIT, self).__init__()
        self.factor = 2 if bilinear else 1
        self.up1    = UpAdaIN(1024, 512 // self.factor, bilinear)
        self.up2    = UpAdaIN(512, 256 // self.factor,  bilinear)
        self.up3    = UpAdaIN(256, 128 // self.factor,  bilinear)
        self.up4    = UpAdaIN(128, 64, bilinear)
        self.outc   = OutConv(64, out_channels)
        # Initiate mlp (predicts AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP_MUNIT(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, x, k, list_x, style_code):
        # Update AdaIN parameters by MLP prediction based off style code
        self.assign_adain_params(self.mlp(style_code))
        x_up_1 = self.up1(x,list_x[0])
        x_up_2 = self.up2(x_up_1,list_x[1])
        x_up_3 = self.up3(x_up_2,list_x[2])
        x_up_4 = self.up4(x_up_3,list_x[3])
        out    = self.outc(x_up_4)
        return out + k
        


##############################
#        Discriminator
##############################


class MultiDiscriminatorMUNIT(nn.Module):
    def __init__(self, in_channels):
        super(MultiDiscriminatorMUNIT, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            #x = self.downsample(x)
        return outputs


'''
Unsupervised Image-to-Image Translation Networks Model
https://arxiv.org/abs/1703.00848
'''


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def reparameterizationBicycleGAN(mu, logvar):
    Tensor = torch.cuda.FloatTensor 
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 8))))
    z = sampled_z * std + mu
    return z


########
# UNIT #
########

class EncoderUNIT(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(EncoderUNIT, self).__init__()

        self.bilinear = bilinear
        self.inc      = DoubleConv(in_channels, 64)
        self.down1    = Down(64, 128)
        self.down2    = Down(128, 256)
        self.down3    = Down(256, 512)
        self.factor   = 2 if bilinear else 1
        self.down4    = Down(512, 1024 // self.factor)

    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        mu = self.down4(x4)
        z  = self.reparameterization(mu)
        return mu, z, [x4,x3,x2,x1]

class GeneratorUNIT(nn.Module):
    def __init__(self, out_channels, bilinear=True):
        super(GeneratorUNIT, self).__init__()

        self.factor = 2 if bilinear else 1
        self.up1    = Up(1024, 512 // self.factor, bilinear)
        self.up2    = Up(512, 256 // self.factor,  bilinear)
        self.up3    = Up(256, 128 // self.factor,  bilinear)
        self.up4    = Up(128, 64, bilinear)
        self.outc   = OutConv(64, out_channels)
        

    def forward(self,x,k,list_x):
        x_up_1 = self.up1(x,list_x[0])
        x_up_2 = self.up2(x_up_1,list_x[1])
        x_up_3 = self.up3(x_up_2,list_x[2])
        x_up_4 = self.up4(x_up_3,list_x[3])
        out = self.outc(x_up_4)
        return k + out

##############################
#        Discriminator
##############################

class DiscriminatorUNIT(nn.Module):
    def __init__(self, input_shape):
        super(DiscriminatorUNIT, self).__init__()
        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 3, padding=1)
        )

    def forward(self, img):
        return self.model(img)


'''
    < ConvBlock >
    Small unit block consists of [convolution layer - normalization layer - non linearity layer]
    
    * Parameters
    1. in_dim : Input dimension(channels number)
    2. out_dim : Output dimension(channels number)
    3. k : Kernel size(filter size)
    4. s : stride
    5. p : padding size
    6. norm : If it is true add Instance Normalization layer, otherwise skip this layer
    7. non_linear : You can choose between 'leaky_relu', 'relu', 'None'
'''

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []
        
        # Convolution Layer
        layers += [nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]
        
        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]
            
        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        
        self.conv_block = nn.Sequential(* layers)
        
    def forward(self, x):
        out = self.conv_block(x)
        return out
    
'''
    < DeonvBlock >
    Small unit block consists of [transpose conv layer - normalization layer - non linearity layer]
    
    * Parameters
    1. in_dim : Input dimension(channels number)
    2. out_dim : Output dimension(channels number)
    3. k : Kernel size(filter size)
    4. s : stride
    5. p : padding size
    6. norm : If it is true add Instance Normalization layer, otherwise skip this layer
    7. non_linear : You can choose between 'relu', 'tanh', None
'''

class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='relu'):
        super(DeconvBlock, self).__init__()
        layers = []
        
        # Transpose Convolution Layer
        layers += [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]
        
        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]
        
        # Non-Linearity Layer
        if non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif non_linear == 'tanh':
            layers += [nn.Tanh()]
            
        self.deconv_block = nn.Sequential(* layers)
            
    def forward(self, x):
        out = self.deconv_block(x)
        return out

'''
    < Generator >
    U-Net Generator. See https://arxiv.org/abs/1505.04597 figure 1 
    or https://arxiv.org/pdf/1611.07004 6.1.1 Generator Architectures
    
    Downsampled activation volume and upsampled activation volume which have same width and height
    make pairs and they are concatenated when upsampling.
    Pairs : (up_1, down_6) (up_2, down_5) (up_3, down_4) (up_4, down_3) (up_5, down_2) (up_6, down_1)
            down_7 doesn't have a partener.
    
    ex) up_1 and down_6 have same size of (N, 512, 2, 2) given that input size is (N, 3, 128, 128).
        When forwarding into upsample_2, up_1 and down_6 are concatenated to make (N, 1024, 2, 2) and then
        upsample_2 makes (N, 512, 4, 4). That is why upsample_2 has 1024 input dimension and 512 output dimension 
        
        Except upsample_1, all the other upsampling blocks do the same thing.
'''


class SimpleGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(SimpleGenerator, self).__init__()
        self.bilinear = bilinear
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)


    def forward(self,x,flag):
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
        x_up_1   = self.up1(x5_noise,x4)
        x_up_2   = self.up2(x_up_1,x3)
        x_up_3   = self.up3(x_up_2,x2)
        x_up_4   = self.up4(x_up_3,x1)
        img      = self.outc(x_up_4)
        return x+img



#####################################################################################

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim=8, bilinear=True):
        super(Generator, self).__init__()
        self.bilinear = bilinear
        self.inc = DoubleConv(in_channels + z_dim, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
    
    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x, z, flag):
        # z : (N, z_dim) -> (N, z_dim, 1, 1) -> (N, z_dim, H, W)
        # x_with_z : (N, 3 + z_dim, H, W)
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat([x, z], dim=1)
        x1       = self.inc(x_with_z)
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
        x_up_1   = self.up1(x5_noise,x4)
        x_up_2   = self.up2(x_up_1,x3)
        x_up_3   = self.up3(x_up_2,x2)
        x_up_4   = self.up4(x_up_3,x1)
        img      = self.outc(x_up_4)
        return x+img

#####################################################################################

class EncoderAB(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(EncoderAB, self).__init__()
        self.bilinear = bilinear
        self.inc      = DoubleConv(in_channels, 64)
        self.down1    = Down(64, 128)
        self.down2    = Down(128, 256)
        self.down3    = Down(256, 512)
        factor        = 2 if bilinear else 1
        self.down4    = Down(512, 1024 // factor)

    def forward(self, x, flag):
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
        return x5_noise, [x4,x3,x2,x1]

class EncoderVAE(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(EncoderVAE, self).__init__()
        self.bilinear = bilinear
        self.inc      = DoubleConv(in_channels, 64)
        self.down1    = Down(64, 128)
        self.down2    = Down(128, 256)
        self.down3    = Down(256, 512)
        factor        = 2 if bilinear else 1
        self.down4    = Down(512, 1024 // factor)

    def forward(self, x):
        x1       = self.inc(x)
        x2       = self.down1(x1)
        x3       = self.down2(x2)
        x4       = self.down3(x3)
        x5       = self.down4(x4)
        return x5, [x4,x3,x2,x1]

class DecoderAB(nn.Module):
    def __init__(self,out_channels,bilinear=True):
        super(DecoderAB, self).__init__()
        factor     = 2 if bilinear else 1
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = OutConv(64, out_channels)

    def forward(self,x,k,list_x):
        x_up_1 = self.up1(x,list_x[0])
        x_up_2 = self.up2(x_up_1,list_x[1])
        x_up_3 = self.up3(x_up_2,list_x[2])
        x_up_4 = self.up4(x_up_3,list_x[3])
        out_net = self.outc(x_up_4)
        return k + out_net


class DecoderBA(nn.Module):
    def __init__(self,out_channels,bilinear=True):
        super(DecoderBA, self).__init__()
        factor     = 2 if bilinear else 1
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = OutConv(64, out_channels)

    def forward(self, x, k, list_x):
        x_up_1 = self.up1(x,list_x[0])
        x_up_2 = self.up2(x_up_1,list_x[1])
        x_up_3 = self.up3(x_up_2,list_x[2])
        x_up_4 = self.up4(x_up_3,list_x[3])
        out_net = self.outc(x_up_4)
        return k + out_net


class EncoderBA(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(EncoderBA, self).__init__()
        self.bilinear = bilinear
        self.inc      = DoubleConv(in_channels, 64)
        self.down1    = Down(64, 128)
        self.down2    = Down(128, 256)
        self.down3    = Down(256, 512)
        factor        = 2 if bilinear else 1
        self.down4    = Down(512, 1024 // factor)

    def forward(self, x, flag):
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
        return x5_noise, [x4,x3,x2,x1]


class GetMeanStd(nn.Module):
    def __init__(self, bilinear=True):
        super(GetMeanStd, self).__init__()
        # Return mu and logvar for reparameterization trick
        self.down1    = Down(512, 128)
        self.down2    = Down(128, 64)
        self.down3    = Down(64, 32)
        self.fc_mu = nn.Linear(32*3*3, 25*25*512)
        self.fc_logvar = nn.Linear(32*3*3, 25*25*512)

    def forward(self, x):
        x = self.down3(self.down2(self.down1(x)))
        out = x.view(x.size(0), -1)
        mu = self.fc_mu(out)
        log_var = self.fc_logvar(out)
        return (mu, log_var)
          


#####################################################################################


'''
    < Discriminator >
    
    PatchGAN discriminator. See https://arxiv.org/pdf/1611.07004 6.1.2 Discriminator architectures.
    It uses two discriminator which have different output sizes(different local probabilities).
    
    Futhermore, it is conditional discriminator so input dimension is 6. You can make input by concatenating
    two images to make pair of Domain A image and Domain B image. 
    There are two cases to concatenate, [Domain_A, Domain_B_ground_truth] and [Domain_A, Domain_B_generated]
    
    d_1 : (N, 6, 128, 128) -> (N, 1, 14, 14)
    d_2 : (N, 6, 128, 128) -> (N, 1, 30, 30)
    
    In training, the generator needs to fool both of d_1 and d_2 and it makes the generator more robust.
 
'''  

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()       
        # Discriminator with last patch (14x14)
        # (N, 6, 128, 128) -> (N, 1, 14, 14)
        self.d_1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=0, count_include_pad=False),
                                 ConvBlock(100, 32, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(32, 64, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(64, 128, k=4, s=1, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 1, k=4, s=1, p=1, norm=False, non_linear=None))
        
    def forward(self, x):
        out_1 = self.d_1(x)
        return out_1
    




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()       
        # Discriminator with last patch (14x14)
        # (N, 6, 128, 128) -> (N, 1, 14, 14)
        self.d_1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=0, count_include_pad=False),
                                 ConvBlock(200, 32, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(32, 64, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(64, 128, k=4, s=1, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 1, k=4, s=1, p=1, norm=False, non_linear=None))
        
        # Discriminator with last patch (30x30)
        # (N, 6, 128, 128) -> (N, 1, 30, 30)
        self.d_2 = nn.Sequential(ConvBlock(200, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(64, 128, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 256, k=4, s=1, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(256, 1, k=4, s=1, p=1, norm=False, non_linear=None))
    
    def forward(self, x):
        out_1 = self.d_1(x)
        out_2 = self.d_2(x)
        return (out_1, out_2)
    
'''
    < ResBlock >
    
    This residual block is different with the one we usaully know which consists of 
    [conv - norm - act - conv - norm] and identity mapping(x -> x) for shortcut.
    
    Also spatial size is decreased by half because of AvgPool2d.
'''

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        
        self.short_cut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out
        
'''
    < Encoder >
    
    Output is mu and log(var) for reparameterization trick used in Variation Auto Encoder.
    Encoding is done in this order.
    1. Use this encoder and get mu and log_var
    2. std = exp(log(var / 2))
    3. random_z = N(0, 1)
    4. encoded_z = random_z * std + mu (Reparameterization trick)
'''

class Encoder(nn.Module):
    def __init__(self, z_dim=8):
        super(Encoder, self).__init__()
        
        self.conv = nn.Conv2d(100, 64, kernel_size=4, stride=2, padding=1)
        self.res_blocks = nn.Sequential(ResBlock(64, 128),
                                        ResBlock(128, 192),
                                        ResBlock(192, 192),
                                        ResBlock(192, 256))
        self.pool_block = nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.AvgPool2d(kernel_size=8, stride=8, padding=0))
        
        # Return mu and logvar for reparameterization trick
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)
        
    def forward(self, x):
        # (N, 3, 128, 128) -> (N, 64, 64, 64)
        out = self.conv(x)
        # (N, 64, 64, 64) -> (N, 128, 32, 32) -> (N, 192, 16, 16) -> (N, 256, 8, 8)
        out = self.res_blocks(out)
        # (N, 256, 8, 8) -> (N, 256, 1, 1)
        out = self.pool_block(out)
        # (N, 256, 1, 1) -> (N, 256)
        out = out.view(x.size(0), -1)
        # (N, 256) -> (N, z_dim) x 2
        mu = self.fc_mu(out)
        log_var = self.fc_logvar(out)
        return (mu, log_var)


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
        return x+0.5*img,logits




class Open(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Open, self).__init__()
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
