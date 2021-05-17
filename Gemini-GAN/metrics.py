
from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
import math
import warnings
warnings.filterwarnings("ignore")

def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    ssim_val = compare_ssim(gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max())
    if  math.isnan(ssim_val) or \
        ssim_val == -np.inf  or \
        ssim_val == np.inf:
            return 0
    else:
            return ssim_val

def getPSNR(gt,pred):
    psnr_list = []
    val = psnr(gt[i],pred[i])
    if  math.isnan(val) or \
        val == -np.inf  or \
        val == np.inf:
        psnr_list.append(0.0)
    else:
        psnr_list.append(val)
    return np.mean(psnr_list)

def huber_loss(x):
    bsize, csize, ssize, height, width = x.size()
    d_x = torch.index_select(x, 4, torch.arange(1, width).cuda())  - torch.index_select(x, 4, torch.arange(width-1).cuda())
    d_y = torch.index_select(x, 3, torch.arange(1, height).cuda()) - torch.index_select(x, 3, torch.arange(height-1).cuda())
    d_z = torch.index_select(x, 2, torch.arange(1, ssize).cuda())  - torch.index_select(x, 2, torch.arange(ssize-1).cuda())
    err = torch.sum(torch.mul(d_x, d_x))/height + torch.sum(torch.mul(d_y, d_y))/width + torch.sum(torch.mul(d_z, d_z))/ssize 
    #err = torch.sum(torch.mul(d_x, d_x))/height + torch.sum(torch.mul(d_y, d_y))/width 
    err /= bsize
    tv_err = torch.sqrt(0.01+err)
    return tv_err
