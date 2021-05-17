#################################################
# Nicolo Savioli, PhD - Imperial College London # 
#################################################

import os
import numpy             as np
import ntpath
import shutil
import cv2
import random
import SimpleITK         as sitk
from   torch.autograd    import Function, Variable
import torch
from   argumentation     import argumentation
import collections
import math
from   collections import OrderedDict
import matplotlib.pyplot as plt
import math
import torch.nn as nn

class plots():

    def write_txt(self,lossList,lossSavePath):
        if os.path.exists(lossSavePath):
            os.remove(lossSavePath) 
        with open(lossSavePath, "a") as f:
            for d in range(len(lossList)):
                f.write(str(lossList[d]) +"\n")

    def psnr_plot(self,psnr_list_valid,pathSave,fr):
        fig            = plt.figure()
        ax             = fig.add_subplot(111)
        # txt write
        self.write_txt (psnr_list_valid,os.path.join(pathSave,"valid_HR_PSNR_"+fr+".txt"))
        ax.set_title   ("PSNR valid")
        ax.plot        (psnr_list_valid, '-',  label="PSNR valid",color='r')
        ax.set_ylabel  ('Peak Signal-To-Noise Ratio')
        ax.set_xlabel  ("Epochs")
        ax.legend      (loc='lower right')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label        = OrderedDict(zip(labels, handles))
        plt.legend     (by_label.values(), by_label.keys())
        fig.savefig    (os.path.join  (pathSave, "HR_PSNR_"+fr+".jpg"))
    

    def psnr_plot_LR_genscan(self,psnr_list_valid,pathSave,fr):
        fig            = plt.figure()
        ax             = fig.add_subplot(111)
        # txt write
        self.write_txt (psnr_list_valid,os.path.join(pathSave,"valid_GenScan_LR_PSNR_"+fr+".txt"))
        ax.set_title   ("PSNR valid")
        ax.plot        (psnr_list_valid, '-',  label="PSNR valid",color='r')
        ax.set_ylabel  ('Peak Signal-To-Noise Ratio Low Resolution')
        ax.set_xlabel  ("Epochs")
        ax.legend      (loc='lower right')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label        = OrderedDict(zip(labels, handles))
        plt.legend     (by_label.values(), by_label.keys())
        fig.savefig    (os.path.join  (pathSave, "GenScan_LR_PSNR_"+fr+".jpg"))


    def psnr_plot_LR_UKBB(self,psnr_list_valid,pathSave,fr):
        fig            = plt.figure()
        ax             = fig.add_subplot(111)
        # txt write
        self.write_txt (psnr_list_valid,os.path.join(pathSave,"valid_UKBB_LR_PSNR_"+fr+".txt"))
        ax.set_title   ("PSNR valid")
        ax.plot        (psnr_list_valid, '-',  label="PSNR valid",color='r')
        ax.set_ylabel  ('Peak Signal-To-Noise Ratio Low Resolution')
        ax.set_xlabel  ("Epochs")
        ax.legend      (loc='lower right')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label        = OrderedDict(zip(labels, handles))
        plt.legend     (by_label.values(), by_label.keys())
        fig.savefig    (os.path.join  (pathSave, "UKBB_LR_PSNR_"+fr+".jpg"))

    def ssim_plot(self,ssim_list_valid,pathSave,fr):
        fig            = plt.figure()
        ax             = fig.add_subplot(111)
        # txt write
        self.write_txt (ssim_list_valid,os.path.join(pathSave,"valid_SSIM_"+fr+".txt"))
        ax.set_title   ("PSNR valid")
        ax.plot        (ssim_list_valid, '-',  label="SSIM valid",color='r')
        ax.set_ylabel  ('Structural Similarity (SSIM)')
        ax.set_xlabel  ("Epochs")
        ax.legend      (loc='lower right')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label        = OrderedDict(zip(labels, handles))
        plt.legend     (by_label.values(), by_label.keys())
        fig.savefig    (os.path.join  (pathSave, "SSIM_"+fr+".jpg"))


    def nmse_plot(self,nmse_list_valid,pathSave):
        fig            = plt.figure()
        ax             = fig.add_subplot(111)
        # txt write
        self.write_txt (nmse_list_valid,os.path.join   (pathSave,"valid_NMSE.txt"))
        ax.set_title   ("NMSE of SuperResMorph")
        ax.plot        (nmse_list_valid, '-',  label="NMSE valid",color='r')
        ax.set_ylabel  ('Normalised Mean Square Error')
        ax.set_xlabel  ("Epochs")
        ax.legend      (loc='lower right')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label        = OrderedDict(zip(labels, handles))
        plt.legend     (by_label.values(), by_label.keys())
        fig.savefig    (os.path.join  (pathSave, "NMSE.jpg"))


    def dice_plot(self,train_list_dice,pathSave,fr):
        fig            = plt.figure()
        ax             = fig.add_subplot(111)
        # txt write
        self.write_txt (train_list_dice,os.path.join(pathSave,"valid_DICE_" + fr + ".txt"))
        ax.set_title   ("DICE valid")
        ax.plot        (train_list_dice, '-',  label="DICE valid",color='r')
        ax.set_ylabel  ('DICE (%)')
        ax.set_xlabel  ("Epochs")
        ax.legend      (loc='lower right')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label        = OrderedDict(zip(labels, handles))
        plt.legend     (by_label.values(), by_label.keys())
        fig.savefig    (os.path.join  (pathSave, "DICE_"+fr+".jpg"))
    

    def write_test_results(self,resultNMSE,resultPSNR,resultDICE,SavePath):
        with open(os.path.join(SavePath,"final_table.txt"), "a") as f:
            f.write("... DICE  mean: "+ str(resultNMSE[0]) +" std: " + str(resultNMSE[1]) + "\n")
            f.write("... PSNR  mean: "+ str(resultPSNR[0]) +" std: " + str(resultPSNR[1]) + "\n")
            f.write("... NMSE  mean: "+ str(resultDICE[0]) +" std: " + str(resultDICE[1]) + "\n")


    def write_test_results_seg(self,resultNMSE,SavePath):
        with open(os.path.join(SavePath,"final_table.txt"), "a") as f:
            f.write("... DICE  mean: "+ str(resultNMSE[0]) +" std: " + str(resultNMSE[1]) + "\n")