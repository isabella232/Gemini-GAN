##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################

import os, glob
from   multiprocessing import Pool
from   functools       import partial
from   tqdm import tqdm
import multiprocessing as mp
import multiprocessing.pool as mpool
from multiprocessing.util import debug
import nibabel as nib
import numpy as np
from shutil import copyfile
import random
import vtk
import shutil


########################################################
# Before starting the code please configure the paths  #
########################################################

###########
# Setting #
####################################
genscan_path = ""
ukbb_path = ""
save_folder = ""
###################################


def check(seg,labels,N=5):
    getlabels = []
    c = True
    for j in range(N):
        if seg[seg == j] != []:
            getlabels.append(j)
        else:
             continue
    if labels in getlabels:
       c = True 
    else:
       c = False
    return c 
    
def landmarks(path, save, labels):
    nim = nib.load(path)
    affine = nim.affine
    seg = nim.get_data()
    if check(seg,labels):
            list_l = []
            z_axis_affine = np.copy(nim.affine[:3, 2])
            for l in labels:
                z = np.nonzero(seg == l)[2]
                z_min, z_max = z.min(), z.max()
                z_mid = int(round(0.5 * (z_min + z_max)))
                if z_axis_affine[2] < 0:
                    z_l = [z_min, z_mid, z_max]
                else:
                    z_l = [z_max, z_mid, z_min]
                for z in  z_l:
                    seg = np.squeeze(seg)
                    x, y = [np.mean(i) for i in np.nonzero(seg[:, :, z] == l)]
                    pdot = np.dot(affine, np.array([x, y, z, 1]).reshape((4, 1)))[:3, 0]
                    list_l.append(pdot)
            k_points = vtk.vtkPoints()
            for pointxyz in list_l:
                k_points.InsertNextPoint(pointxyz[0], pointxyz[1], [2])
            poly = vtk.vtkPolyData()
            poly.SetPoints(k_points)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(poly)
            writer.SetFileName(save)
            writer.Write()

def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path


def fixLabel(seg_path):
    nim_img = nib.load(seg_path)
    seg = nim_img.get_data()    
    seg[seg[:,:,:]==4]=3
    affine = nim_img.affine
    nim2 = nib.Nifti1Image(seg, affine)
    nib.save(nim2, seg_path)
    os.system('chmod 777 ' + seg_path)


def getEpi(seg_path,seg_new):
    nim_img = nib.load(seg_path)
    seg = nim_img.get_data() 
    if len(seg.shape) == 4:
        new_seg = np.zeros((seg.shape[0],seg.shape[1],seg.shape[2],seg.shape[3]))
        new_seg[seg[:,:,:,:]==2]=1
    else:
        new_seg = np.zeros((seg.shape[0],seg.shape[1],seg.shape[2]))
        new_seg[seg[:,:,:]==2]=1
   
    affine = nim_img.affine
    nim2 = nib.Nifti1Image(new_seg, affine)
    nib.save(nim2, seg_new)

def GetAffine(template_path):
    nim_temp = nib.load(template_path)
    na = np.zeros((4,4))
    a11 =1.25
    a14 = nim_temp.affine[0][3]
    a22 =1.25
    a24 = nim_temp.affine[1][3]
    a32 = 2.0 
    a34 = nim_temp.affine[2][3]
    a44 = nim_temp.affine[3][3]
    na[0][0] = a11
    na[0][3] = a14 
    na[1][1] = a22
    na[1][3] = a24
    na[2][2] = a32
    na[2][3] = a34
    na[3][3] = a44
    return na
    
ukbb_list = os.listdir(ukbb_path)
lenght_ukbb = len(ukbb_list)

for dir_name in tqdm(os.listdir(genscan_path)):
    j_ukbb = random.randint(0,lenght_ukbb-1)
    ukbb_patient_path = os.path.join(ukbb_path,ukbb_list[j_ukbb]) 
    genscan_patient_path = os.path.join(genscan_path,dir_name) 
    save_path = mkdir_dir(os.path.join(save_folder,dir_name))
    save_path_temp = mkdir_dir(os.path.join(save_path,"tmp"))

    for fr in ["ED","ES"]:

        gray_genscan = os.path.join(genscan_patient_path,"low_gray_genscan_"+fr+".nii.gz")
        seg_genscan = os.path.join(genscan_patient_path,"hight_seg_genscan_"+fr+".nii.gz")
        seg_genscan_save = os.path.join(save_path,"hight_seg_genscan_"+fr+".nii.gz")
        seg_genscan_low = os.path.join(genscan_patient_path,"low_seg_genscan_"+fr+".nii.gz")
        seg_genscan_low_save = os.path.join(save_path,"low_seg_genscan_"+fr+".nii.gz")
        high_grayscale_genscan = os.path.join(genscan_patient_path,"hight_gray_genscan_"+fr+".nii.gz")
        high_grayscale_genscan_save = os.path.join(save_path,"hight_gray_genscan_"+fr+".nii.gz")
        gray_ukbb = os.path.join(ukbb_patient_path,"lvsa_SR_"+fr+".nii.gz")
        seg_ukbb = os.path.join(ukbb_patient_path,"seg_lvsa_SR_"+fr+".nii.gz")
        seg_ukbb_low = os.path.join(ukbb_patient_path,"LVSA_seg_"+fr+".nii.gz")
        seg_ukbb_low_save = os.path.join(save_path,"low_seg_ukbb_"+fr+".nii.gz")
        landmarks_ukbb_path = os.path.join(save_path_temp,"landmarks_ukbb_"+fr+".vtk")
        landmarks_genscan_path = os.path.join(save_path_temp,"landmarks_genscan_"+fr+".vtk")
        landmarks_genscan_low_path = os.path.join(save_path_temp,"landmarks_genscan_low_"+fr+".vtk")
        landmarks_genscan_hight_path = os.path.join(save_path_temp,"landmarks_genscan_hight_"+fr+".vtk")
        gray_genscan_save = os.path.join(save_path,"low_gray_genscan_"+fr+".nii.gz")
        gray_ukbb_save = os.path.join(save_path,"low_gray_ukbb_"+fr+".nii.gz")
        seg_ukbb_save = os.path.join(save_path,"hight_seg_ukbb_"+fr+".nii.gz")
        gray_ukbb_save = os.path.join(save_path,"low_gray_ukbb_"+fr+".nii.gz")
        seg_ukbb_save = os.path.join(save_path,"hight_seg_ukbb_"+fr+".nii.gz")
        
        # Hight resolution and low resolution alignment
        fixLabel(seg_genscan)
        fixLabel(seg_genscan_low)
        
        # landmarks
        landmarks(seg_genscan,landmarks_genscan_path,labels = [2,3])
        landmarks(seg_ukbb,landmarks_ukbb_path,labels = [2,3])

        os.system('prreg '
            '{0} '
            '{1} '
            '-dofout {2}/landmarks.dof.gz >/dev/nul '
            .format(landmarks_ukbb_path, landmarks_genscan_path, save_path_temp))

        # Alignment HR GRAY GenScan --> UKBB
        os.system('mirtk transform-image  '
            '{0} '
            '{1} '
            '-dofin {2}/landmarks.dof.gz '
            '-target {3} -interp BSpline >/dev/nu' 
            .format(high_grayscale_genscan, high_grayscale_genscan_save, save_path_temp, seg_ukbb))
      
        # Alignment LR GRAY GenScan --> UKBB
        os.system('mirtk transform-image  '
            '{0} '
            '{1} '
            '-dofin {2}/landmarks.dof.gz '
            '-target {3} -interp BSpline >/dev/nu' 
            .format(gray_genscan, gray_genscan_save, save_path_temp, gray_ukbb))
        
        # Alignment HR SEG GenScan --> UKBB
        os.system('mirtk transform-image  '
            '{0} '
            '{1} '
            '-dofin {2}/landmarks.dof.gz '
            '-target {3} -interp NN >/dev/nu' 
            .format(seg_genscan, seg_genscan_save, save_path_temp, seg_ukbb))
    
        # Alignment LR SEG GenScan --> UKBB
        os.system('mirtk transform-image  '
            '{0} '
            '{1} '
            '-dofin {2}/landmarks.dof.gz '
            '-target {3} -interp NN >/dev/nu' 
            .format(seg_genscan_low, seg_genscan_low_save, save_path_temp, seg_ukbb))
        
        copyfile(gray_ukbb, gray_ukbb_save)
        copyfile(seg_ukbb, seg_ukbb_save)
        copyfile(seg_ukbb_low, seg_ukbb_low_save)
        
    os.system('chmod -R 777 ' + save_folder)
    shutil.rmtree(save_path_temp)






