############################################
# Nicolo Savioli, Imperial Collage London  # 
############################################

import torch.utils.data as data
import numpy            as np
import datetime
import os
import numpy             as np
import ntpath
import shutil
import cv2
import random
from   torch.autograd    import Function, Variable
import torch
from   argumentation     import argumentation
import torch.nn as nn
from   loadCardiacData    import LoadTrainDataset

class TrainDataset(data.Dataset):
    def __init__(self,pathData,pathCode,n_output,typedata):
        super(TrainDataset, self).__init__()
        self.getdata  = LoadTrainDataset(pathData,pathCode,n_output,typedata)

    def __getitem__(self, index):        
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        # load numpy array 
        img,ground_truth,gt_ukbb = self.getdata.getLoadTrain(index)
        ########################################################
        return img,ground_truth,gt_ukbb

    def __len__(self):
        return self.getdata.len_data()

