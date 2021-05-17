###################
# Nicolo Savioli  #
###################

import os 
from   shutil import copyfile
import SimpleITK  as sitk
import numpy as np 
from   operator import truediv
from   random import randint
import secrets 
import numpy as np, nibabel as nib
import pickle
import csv
import pandas as pd  
import shutil
from   os import path
from   tqdm import tqdm

def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def mul_list(new_a,new_b):
    return tuple([a*b for a,b in zip(new_a,new_b)])

def div_list(new_a,new_b):
    return tuple([a/b for a,b in zip(new_a,new_b)])

def GetNewSpacing(origin_size,new_size,origin_spacing):
    origin_size     = list(origin_size)
    new_size        = list(new_size)
    origin_spacing  = list(origin_spacing)
    new_spacing     = mul_list(origin_size, div_list(origin_spacing,new_size))
    return tuple(new_spacing)

def ResizeVolumeLinear(original,size_data):
    new_spacing      = GetNewSpacing(original.GetSize(),size_data,original.GetSpacing())
    new              = sitk.Resample(original, size_data,
                                    sitk.Transform(), 
                                    sitk.sitkLinear,
                                    original.GetOrigin(),
                                    new_spacing,
                                    original.GetDirection(),
                                    0,
                                    original.GetPixelID())
    return new


def ResizeVolumeLinearkBSpline(original,size_data):
    new_spacing      = GetNewSpacing(original.GetSize(),size_data,original.GetSpacing())
    new              = sitk.Resample(original, size_data,
                                    sitk.Transform(), 
                                    sitk.sitkBSpline,
                                    original.GetOrigin(),
                                    new_spacing,
                                    original.GetDirection(),
                                    0,
                                    original.GetPixelID())
    return new



def makedataset(path_open,path_save):
    for patient in tqdm(os.listdir(path_open)): 
        save_folder = os.path.join(path_save,patient)
        mkdir_dir(save_folder)
        data_path = os.path.join(path_open,patient)
        for fd in ["ED","ES"]:
            ##################################################################################################
            # low genscan grayscale 
            low_gray_genscan         = sitk.ReadImage(os.path.join(data_path,"final_data","GENSCAN_SYN_"+fd+".nii.gz"))
            # hight genscan grayscale 
            hight_gray_genscan       = sitk.ReadImage(os.path.join(data_path,"final_data","GENSCAN_NO_SYN_"+fd+".nii.gz"))
            hight_gray_genscan       = ResizeVolumeLinear(hight_gray_genscan,low_gray_genscan.GetSize())
            # low segmentation genscan 
            low_seg_genscan          = sitk.ReadImage(os.path.join(data_path,"final_data","GENSCAN_LOW_SEG_"+fd+".nii.gz"))
            # high segmentation genscan 
            high_seg_genscan         = sitk.ReadImage(os.path.join(data_path,"final_data","GENSCAN_HIGH_SEG_"+fd+".nii.gz"))
            # low ukbb grayscale 
            low_gray_ukbb            = sitk.ReadImage(os.path.join(data_path,"final_data","UKBB_SYN_" + fd + ".nii.gz"))
            # test interpolation 
            low_gray_genscan_bspline = ResizeVolumeLinearkBSpline(low_gray_genscan,low_gray_genscan.GetSize())
            ##################################################################################################
            sitk.WriteImage(low_gray_genscan,os.path.join(save_folder,"low_gray_genscan_" + fd + ".nii.gz"))
            sitk.WriteImage(hight_gray_genscan,os.path.join(save_folder,"hight_gray_genscan_" + fd + ".nii.gz"))
            sitk.WriteImage(low_seg_genscan,os.path.join(save_folder,"low_seg_genscan_" + fd + ".nii.gz"))
            sitk.WriteImage(high_seg_genscan,os.path.join(save_folder,"hight_seg_genscan_" + fd +".nii.gz"))
            sitk.WriteImage(low_gray_ukbb,os.path.join(save_folder,"low_gray_ukbb_" + fd + ".nii.gz"))
            sitk.WriteImage(low_gray_genscan_bspline,os.path.join(save_folder,"test_interp_bspline_" + fd + ".nii.gz"))
            ##################################################################################################

if __name__ == "__main__":
    data_folder = "/mnt/storage/home/nsavioli/cardiac/4D_superesolution_data/SUPERES_DATASET"
    save_folder = "/mnt/storage/home/nsavioli/cardiac/4D_superesolution_data/genscan_dataset"
    makedataset(data_folder,save_folder)

