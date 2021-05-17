############################################
# Nicolo Savioli, Imperial Collage London  # 
############################################


from   torch.autograd    import Function, Variable
import torch
import torch.nn as     nn
import torch
from   torch.utils.data import DataLoader

def modelPretrain(generator_gan,generator_pretrain):
    with torch.no_grad():
        generator_gan.inc.double_conv[0].weight = generator_pretrain.inc.double_conv[0].weight
        generator_gan.inc.double_conv[1].weight = generator_pretrain.inc.double_conv[1].weight
        generator_gan.inc.double_conv[3].weight = generator_pretrain.inc.double_conv[3].weight
        generator_gan.inc.double_conv[4].weight = generator_pretrain.inc.double_conv[4].weight
        generator_gan.down1.maxpool_conv[1].double_conv[0].weight = generator_pretrain.down1.maxpool_conv[1].double_conv[0].weight
        generator_gan.down1.maxpool_conv[1].double_conv[1].weight = generator_pretrain.down1.maxpool_conv[1].double_conv[1].weight 
        generator_gan.down1.maxpool_conv[1].double_conv[3].weight = generator_pretrain.down1.maxpool_conv[1].double_conv[3].weight 
        generator_gan.down1.maxpool_conv[1].double_conv[4].weight = generator_pretrain.down1.maxpool_conv[1].double_conv[4].weight 
        generator_gan.down2.maxpool_conv[1].double_conv[0].weight = generator_pretrain.down2.maxpool_conv[1].double_conv[0].weight
        generator_gan.down2.maxpool_conv[1].double_conv[1].weight = generator_pretrain.down2.maxpool_conv[1].double_conv[1].weight 
        generator_gan.down2.maxpool_conv[1].double_conv[3].weight = generator_pretrain.down2.maxpool_conv[1].double_conv[3].weight 
        generator_gan.down2.maxpool_conv[1].double_conv[4].weight = generator_pretrain.down2.maxpool_conv[1].double_conv[4].weight 
        generator_gan.down3.maxpool_conv[1].double_conv[0].weight = generator_pretrain.down3.maxpool_conv[1].double_conv[0].weight
        generator_gan.down3.maxpool_conv[1].double_conv[1].weight = generator_pretrain.down3.maxpool_conv[1].double_conv[1].weight 
        generator_gan.down3.maxpool_conv[1].double_conv[3].weight = generator_pretrain.down3.maxpool_conv[1].double_conv[3].weight 
        generator_gan.down3.maxpool_conv[1].double_conv[4].weight = generator_pretrain.down3.maxpool_conv[1].double_conv[4].weight 
        generator_gan.down4.maxpool_conv[1].double_conv[0].weight = generator_pretrain.down4.maxpool_conv[1].double_conv[0].weight
        generator_gan.down4.maxpool_conv[1].double_conv[1].weight = generator_pretrain.down4.maxpool_conv[1].double_conv[1].weight 
        generator_gan.down4.maxpool_conv[1].double_conv[3].weight = generator_pretrain.down4.maxpool_conv[1].double_conv[3].weight 
        generator_gan.down4.maxpool_conv[1].double_conv[4].weight = generator_pretrain.down4.maxpool_conv[1].double_conv[4].weight 
        generator_gan.up_img_1.conv.double_conv[0].weight = generator_pretrain.up_img_1.conv.double_conv[0].weight
        generator_gan.up_img_1.conv.double_conv[1].weight = generator_pretrain.up_img_1.conv.double_conv[1].weight
        generator_gan.up_img_1.conv.double_conv[3].weight = generator_pretrain.up_img_1.conv.double_conv[3].weight
        generator_gan.up_img_1.conv.double_conv[4].weight = generator_pretrain.up_img_1.conv.double_conv[4].weight
        generator_gan.up_img_2.conv.double_conv[0].weight = generator_pretrain.up_img_2.conv.double_conv[0].weight
        generator_gan.up_img_2.conv.double_conv[1].weight = generator_pretrain.up_img_2.conv.double_conv[1].weight
        generator_gan.up_img_2.conv.double_conv[3].weight = generator_pretrain.up_img_2.conv.double_conv[3].weight
        generator_gan.up_img_2.conv.double_conv[4].weight = generator_pretrain.up_img_2.conv.double_conv[4].weight
        generator_gan.up_img_3.conv.double_conv[0].weight = generator_pretrain.up_img_3.conv.double_conv[0].weight
        generator_gan.up_img_3.conv.double_conv[1].weight = generator_pretrain.up_img_3.conv.double_conv[1].weight
        generator_gan.up_img_3.conv.double_conv[3].weight = generator_pretrain.up_img_3.conv.double_conv[3].weight
        generator_gan.up_img_3.conv.double_conv[4].weight = generator_pretrain.up_img_3.conv.double_conv[4].weight
        generator_gan.up_img_4.conv.double_conv[0].weight = generator_pretrain.up_img_4.conv.double_conv[0].weight
        generator_gan.up_img_4.conv.double_conv[1].weight = generator_pretrain.up_img_4.conv.double_conv[1].weight
        generator_gan.up_img_4.conv.double_conv[3].weight = generator_pretrain.up_img_4.conv.double_conv[3].weight
        generator_gan.up_img_4.conv.double_conv[4].weight = generator_pretrain.up_img_4.conv.double_conv[4].weight
        generator_gan.outc_img.conv = generator_pretrain.outc_img.conv
        generator_gan.up_seg_1.conv.double_conv[0].weight = generator_pretrain.up_seg_1.conv.double_conv[0].weight
        generator_gan.up_seg_1.conv.double_conv[1].weight = generator_pretrain.up_seg_1.conv.double_conv[1].weight
        generator_gan.up_seg_1.conv.double_conv[3].weight = generator_pretrain.up_seg_1.conv.double_conv[3].weight
        generator_gan.up_seg_1.conv.double_conv[4].weight = generator_pretrain.up_seg_1.conv.double_conv[4].weight
        generator_gan.up_seg_2.conv.double_conv[0].weight = generator_pretrain.up_seg_2.conv.double_conv[0].weight
        generator_gan.up_seg_2.conv.double_conv[1].weight = generator_pretrain.up_seg_2.conv.double_conv[1].weight
        generator_gan.up_seg_2.conv.double_conv[3].weight = generator_pretrain.up_seg_2.conv.double_conv[3].weight
        generator_gan.up_seg_2.conv.double_conv[4].weight = generator_pretrain.up_seg_2.conv.double_conv[4].weight
        generator_gan.up_seg_3.conv.double_conv[0].weight = generator_pretrain.up_seg_3.conv.double_conv[0].weight
        generator_gan.up_seg_3.conv.double_conv[1].weight = generator_pretrain.up_seg_3.conv.double_conv[1].weight
        generator_gan.up_seg_3.conv.double_conv[3].weight = generator_pretrain.up_seg_3.conv.double_conv[3].weight
        generator_gan.up_seg_3.conv.double_conv[4].weight = generator_pretrain.up_seg_3.conv.double_conv[4].weight
        generator_gan.up_seg_4.conv.double_conv[0].weight = generator_pretrain.up_seg_4.conv.double_conv[0].weight
        generator_gan.up_seg_4.conv.double_conv[1].weight = generator_pretrain.up_seg_4.conv.double_conv[1].weight
        generator_gan.up_seg_4.conv.double_conv[3].weight = generator_pretrain.up_seg_4.conv.double_conv[3].weight
        generator_gan.up_seg_4.conv.double_conv[4].weight = generator_pretrain.up_seg_4.conv.double_conv[4].weight
        generator_gan.outc_seg.conv = generator_pretrain.outc_seg.conv
    return generator_gan
