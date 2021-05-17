############################################
# Nicolo Savioli, Imperial Collage London  # 
############################################

import SimpleITK         as sitk
from   argumentation     import argumentation
import numpy            as np
import datetime
import os
import torch
import random
from   torchvision import transforms
import cv2
from preprocessing import preprocessing
np.seterr(divide='ignore', invalid='ignore')

class LoadTrainDataset():
    def __init__(self,pathData,pathCode,n_output,typedataset):
        self.typedataset = typedataset
        if typedataset == "SG":
            self.trainData   = self.open_txt(os.path.join(pathCode,"dataset_txt","DA","train.txt"))
        else:
            self.trainData   = self.open_txt(os.path.join(pathCode,"dataset_txt","train.txt"))
        self.pathData    = pathData
        self.pathCode    = pathCode
        self.agrum       = argumentation(n_output)
        self.pre         = preprocessing(400,n_output)
        
    def makedir(self,path):
        if not os.path.exists(path):
            os.mkdir(path)
        return path
        
    def open_txt(self,path):
        data = []
        with open(path, 'r') as f:
            data = [line.strip() for line in f]
        return data
    
    def len_data(self):
        return len(self.trainData)


    def getLoadTrain(self,j):
        index = random.randint(0,1)
        img,ground_truth,lr_sg = None,None,None
        if index == 0:
            if self.typedataset == "UKBB":
                img  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_gray_ukbb_ED.nii.gz")))
            elif self.typedataset == "SG":
                img  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_gray_SG_ED.nii.gz")))
            # ground_truth
            ground_truth  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_gray_genscan_ED.nii.gz")))
            #################################################################################################################################
            if self.typedataset == "UKBB":
                lr_sg  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_seg_ukbb_ED.nii.gz")))
            elif self.typedataset == "SG":
                lr_sg  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_seg_SG_ED.nii.gz")))
                lr_sg  = lr_sg.astype(np.float)

        elif index == 1:
            if self.typedataset == "UKBB":
                img  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_gray_ukbb_ES.nii.gz")))
            elif self.typedataset == "SG":
                img  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_gray_SG_ES.nii.gz")))
            # ground_truth
            ground_truth  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_gray_genscan_ES.nii.gz")))
            #################################################################################################################################
            if self.typedataset == "UKBB":
                lr_sg  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_seg_ukbb_ES.nii.gz")))
            elif self.typedataset == "SG":
                lr_sg  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_seg_SG_ES.nii.gz")))
                lr_sg  = lr_sg.astype(np.float)
                
        tag_img = self.pre.pad_seq(img)
        tag_ground_truth = self.pre.pad_seq(ground_truth)
        tag_sg = self.pre.pad_seq_seg(lr_sg)
        out_tag_img,out_tag_ground_truth,out_tag_sg = self.agrum.get_argum(tag_img,tag_ground_truth,tag_sg)
        return out_tag_img,out_tag_ground_truth,out_tag_sg

