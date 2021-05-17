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
    def __init__(self,pathData,pathCode,n_output):
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
        index          = random.randint(0,1)
        input_img,\
        output_img,\
        output_seg    = None,None,None
        if index == 0:
            # imgs 
            input_img     = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_gray_genscan_ED.nii.gz")))
            output_img    = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"hight_gray_genscan_ED.nii.gz")))
            # seg 
            output_seg    = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"hight_seg_genscan_ED.nii.gz")))
        elif index == 1:
            # imgs 
            input_img     = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"low_gray_genscan_ES.nii.gz")))
            output_img    = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"hight_gray_genscan_ES.nii.gz")))
            # seg 
            output_seg    = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.pathData,self.trainData[j],"hight_seg_genscan_ES.nii.gz")))
        input_img_out     = self.pre.pad_seq(input_img)
        output_img_out    = self.pre.pad_seq(output_img)
        output_seg_out    = self.pre.pad_seq_seg(output_seg)
        arg_input_img_out,\
        arg_output_img_out,\
        arg_output_seg_out = self.agrum.get_argum(input_img_out,output_img_out,output_seg_out)
        return arg_input_img_out,\
               arg_output_img_out,\
               arg_output_seg_out

