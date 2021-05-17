#################################################
# Nicolo Savioli, PhD - Imperial College London # 
#################################################

import numpy as np
import cv2
from   random import randint
import random
from   torch.autograd import Function, Variable 
import torch
import torch.nn as nn
import os 
from   preprocessing import preprocessing
import SimpleITK as sitk
import math
import pandas as pd 

class argumentation():
    def __init__(self,n_output):
        self.crop_factor  = 2
        self.device       = torch.device("cuda")
        self.pre          = preprocessing(400,n_output)
        self.directions   = True


    def removeNaN(self,x):
        if np.isnan(np.max(x)):
            return np.zeros(x.shape,dtype=np.float32)
        else:
            return x

    def unitNormalisation(self,v):        
        v_min = np.amin(v)
        v_max = np.amax(v)
        out   = (v - v_min)/(v_max - v_min)
        return out  

    def convert_uint8(self,img):
        # img is a numpy array/cv2 image
        img = img - img.min() # Now between 0 and 8674
        img = img / img.max() * 255
        return np.uint8(img)

    def random_brightness(self,img):
        alpha = np.random.uniform(0.0,9.0)
        beta  = np.random.uniform(-0.5,0.5)
        nimg  = np.zeros(img.shape)
        for z in range(img.shape[0]):
            nimg[z,:,:] = img[z,:,:]*alpha + beta
        return  nimg

    def flip(self,data,n):
        data_list = []
        for i in range(data.shape[0]):
            data_list.append(cv2.flip(data[i],n))
        return np.asarray(data_list)
    

    def randomCrop(self,img1, img2, mask, width, height):
        x        = random.randint(0, img1.shape[1] - width)
        y        = random.randint(0, img1.shape[2] - height)
        acc_img1 = []
        acc_img2 = []
        acc_mask = []
        for i in range(img1.shape[0]):
            img1_crop = img1[i, y:y+height, x:x+width]
            img2_crop = img2[i, y:y+height, x:x+width]
            mask_crop = mask[i, y:y+height, x:x+width]
            acc_img1.append(img1_crop)
            acc_img2.append(img2_crop)
            acc_mask.append(mask_crop)
        return self.pre.padImg(np.asarray(acc_img1)),\
               self.pre.padImg(np.asarray(acc_img2)),\
               self.pre.padImg(np.asarray(acc_mask))
    
    def getChoice(self,low,hight,seg_hight,random_index):
        low_out = hight_out = seg = None      
        if random_index == 0:
            low_out,\
            hight_out,\
            seg               = self.randomCrop(low,hight,seg_hight,150,150)
        if random_index == 1:
            low_out           = self.random_brightness(low)
            hight_out         = hight
            seg               = seg_hight
        if random_index == 2:
            low_out           = low
            hight_out         = hight
            seg               = seg_hight
        if random_index == 3:
            low_out           = self.flip(low,0)
            hight_out         = self.flip(hight,0)
            seg               = self.flip(seg_hight,0)
        if random_index == 4:
            low_out           = self.flip(low,1)
            hight_out         = self.flip(hight,1)
            seg               = self.flip(seg_hight,1)
        return low_out,\
               hight_out,seg     
    
    def get_argum(self,low,hight,seg):
        list_seg      = []
        list_high     = []
        ################################
        index  = random.randint (0,4) 
        ################################
        arug_low,\
        arug_hight,\
        arug_seg      = self.getChoice(low,hight,seg,index)
        return self.pre.removeNaN(self.pre.unitNormalisation(arug_low)),\
               self.pre.removeNaN(self.pre.unitNormalisation(arug_hight)),arug_seg
        

def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)












