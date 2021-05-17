##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################

from   torch.autograd    import Function, Variable
import torch
from   model import GeneratorUnetSeg,GeneratorUnetImg,GeneratorSRGANSegImg,GeneratorSRGANImg,GeneratorSRGANImg,GeneratorSRGANSeg,GeneratorSRGANSegImgDouble,GeneratorUnetSegImg
import numpy as np 
import torch.cuda as cutorch
import warnings
import torch.nn as     nn
import torch
from   torch.utils.data import DataLoader
from   tqdm import tqdm
import cv2,os
import SimpleITK  as sitk
with   warnings.catch_warnings():
       warnings.filterwarnings("ignore")
from   metrics import * 
from   preprocessing import preprocessing
import nibabel as nib
from   datetime import date

device = torch.device("cuda")

def readVol(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def segPredict(pred_seg):
    _, pred_seg_max      = torch.max (pred_seg, dim=1)
    out = pred_seg_max.long().cpu().data.numpy()
    return out

def load(name_G,path_model,in_channels,out_channels):
    if name_G   == "unet_seg":
        generator     = GeneratorUnetSeg(in_channels,out_channels).cuda()
        generator.load_state_dict(torch.load(path_model))
        generator     = generator.cuda()
        generator.eval()
    elif name_G == "unet_img":
        generator     = GeneratorUnetImg(in_channels,out_channels).cuda()
        generator.load_state_dict(torch.load(path_model))
        generator     = generator.cuda()
        generator.eval()
    elif name_G == "joint_unet" or name_G == "joint_unet_gan":
        generator     = GeneratorUnetSegImg(in_channels,out_channels).cuda()
        generator.load_state_dict(torch.load(path_model))
        generator.eval()
        generator     = generator.cuda()
    elif name_G == "joint_SR_gan":
        generator     = GeneratorSRGANSegImg(in_channels,out_channels,4).cuda()
        generator.load_state_dict(torch.load(path_model))
        generator     = generator.cuda()
        generator.eval()
    elif name_G == "img_SR":
        generator = GeneratorSRGANImg(in_channels,out_channels,4)
        generator.load_state_dict(torch.load(path_model))
        generator = generator.cuda()
        generator.eval()
    elif name_G == "seg_SR":
        generator = GeneratorSRGANSeg(in_channels,out_channels,4)
        generator.load_state_dict(torch.load(path_model))
        generator = generator.cuda()
        generator.eval()
    elif name_G == "img_SR_gan":
        generator = GeneratorSRGANImg(in_channels,out_channels,4)
        generator.load_state_dict(torch.load(path_model))
        generator = generator.cuda()
        generator.eval()
    elif name_G == "seg_SR_gan":
        generator = GeneratorSRGANSeg(in_channels,out_channels,4)
        generator.load_state_dict(torch.load(path_model))
        generator = generator.cuda()
        generator.eval()
    elif name_G == "joint_double_SR_gan":
        generator = GeneratorSRGANSegImgDouble(in_channels,out_channels,4)
        generator.load_state_dict(torch.load(path_model))
        generator = generator.cuda()
        generator.eval()
    return generator

def open_txt(path):
    data = []
    with open(path, 'r') as f:
        data = [line.strip() for line in f]
    return data

def get_torch(data):
    data_torch  = torch.from_numpy(data)
    data_torch[data_torch != data_torch] = 0
    torch_out = data_torch.unsqueeze(0).float()
    return torch_out

def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def getDice(gt,seg,label):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt==label, seg==label)
    return 100*overlap_measures_filter.GetDiceCoefficient()

def CopyInfo(gt,seg):
    gt_img       = sitk.GetArrayFromImage(gt)
    seg_img      = sitk.GetArrayFromImage(seg)
    collect_data = []
    for i in range(gt_img.shape[0]):
        collect_data.append(seg_img[i])
    seg_data = np.asarray(collect_data)
    seg_stik = sitk.GetImageFromArray(seg_data)
    seg_stik.CopyInformation(gt)
    return seg_stik 
    
def globaDice(gt,seg):
    seg  = CopyInfo(gt,seg)
    endo = getDice(gt,seg,1)
    myo  = getDice(gt,seg,2)
    rv   = getDice(gt,seg,3)
    return (endo+myo+rv)/3

def fixSeg(seg):
    seg[seg[:,:,:]==4]=3
    return seg

def unitNormalisation(v):        
    v_min = np.amin(v)
    v_max = np.amax(v)
    out   = (v - v_min)/(v_max - v_min)
    return out  

def PSNR(gt,img,seg_gt):
    psnr_data = []
    for i in range(img.shape[0]):
        if np.mean(seg_gt[i])>0:
            gt_img = unitNormalisation(gt[i])
            pred_img = unitNormalisation(img[i])
            out_psnr = compare_psnr(gt_img,pred_img)
            if  not  math.isnan(out_psnr) or out_psnr == -np.inf  or out_psnr == np.inf or out_psnr == 0 or out_psnr<0:
                psnr_data.append(out_psnr)
    return [np.mean(psnr_data),np.std(psnr_data)]

def SSIM(gt,img,seg_gt):
    ssim_data = []
    for i in range(img.shape[0]):
        if np.mean(seg_gt[i])>0:
            out_ssim = compare_ssim(gt[i],img[i], multichannel=False, data_range=gt.max())
            if  math.isnan(out_ssim) or out_ssim == -np.inf or out_ssim == np.inf:
                continue
            ssim_data.append(out_ssim)
    return [np.mean(ssim_data),np.std(ssim_data)]

    
def calDice(seg,gt,k):
    return np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))

def fix_labels(seg):
    seg[seg[:,:,:]==4]=3
    return seg

'''
def DiceEval(seg,gt):
    seg = fix_labels(seg)
    gt  = fix_labels(gt)
    endo_list,myo_list,rv_list = [],[],[]
    for i in range(seg.shape[0]):
        if np.mean(seg[i])>0 and np.mean(gt[i])>0:
            endo = calDice(seg[i],gt[i],1)
            myo  = calDice(seg[i],gt[i],2)
            rv   = calDice(seg[i],gt[i],3)
            if np.isnan(endo): continue 
            if np.isnan(myo):  continue 
            if np.isnan(rv):   continue 
            if endo == 0: continue
            if myo == 0: continue
            if rv == 0: continue
            endo_list.append(endo)
            myo_list.append(myo)
            rv_list.append(rv)
    return [np.mean(endo_list),np.std(endo_list)],\
        [np.mean(myo_list),np.std(myo_list)],\
        [np.mean(rv_list),np.std(rv_list)]    
'''


def DiceEval(seg,gt):
    seg = fix_labels(seg)
    gt  = fix_labels(gt)
    endo_list,myo_list,rv_list = [],[],[]
    flag = False
    for i in range(seg.shape[0]):
        if np.mean(seg[i])>0 and np.mean(gt[i])>0:
            endo = calDice(seg[i],gt[i],1)
            myo  = calDice(seg[i],gt[i],2)
            rv   = calDice(seg[i],gt[i],3)
            if np.isnan(endo): continue 
            if np.isnan(myo):  continue 
            if np.isnan(rv):   continue 
            if endo == 0: continue
            if myo == 0: continue
            if rv == 0: continue
            endo_list.append(endo)
            myo_list.append(myo)
            rv_list.append(rv)
            flag = True
        if flag == False and i==seg.shape[0]-1:
            endo_list.append(0.0)
            myo_list.append(0.0)
            rv_list.append(0.0)

    return [np.mean(endo_list),np.std(endo_list)],\
           [np.mean(myo_list),np.std(myo_list)],\
           [np.mean(rv_list),np.std(rv_list)]    

def removeNaN(listd):
    cleanedList = [x for x in listd if str(x) != 'nan']
    return cleanedList




