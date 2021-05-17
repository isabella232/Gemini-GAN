
############################################
# Nicolo Savioli, Imperial Collage London  # 
############################################


import SimpleITK         as sitk
import numpy            as np
import datetime
import os
import torch
import random
from   torchvision import transforms
import cv2
from   tqdm import tqdm
import nibabel as nib
import os
import cv2
np.seterr(divide='ignore', invalid='ignore')


class preprocessing():
    def __init__(self,max_size_xy,max_pad_z):
        self.max_pad_xy  = max_size_xy
        self.max_pad_z   = max_pad_z
        self.mean        = 0.12518407280054672
        self.std         = 0.0022657211935882236

    def removeNaN(self,x):
        if np.isnan(np.max(x)):
            return np.zeros(x.shape,dtype=np.float64)
        else:
            return x

    def unitNormalisation(self,v):        
        v_min = np.amin(v)
        v_max = np.amax(v)
        out   = (v - v_min)/(v_max - v_min)
        return out  

    def NormalisationVolume(self,data):
        acc_data = []
        for i in range(data.shape[0]):
            image        = self.removeNaN(self.unitNormalisation(data[i]))
            norm         = np.divide(np.subtract(image, self.mean),self.std)
            acc_data.append(norm)
        return np.asarray(acc_data)

    def mkdir_dir(self,file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        return file_path
    
    def crop_center(self,img,cropx,cropy):
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]

    def padImg(self,image):
        Z   = image.shape[0]
        X   = image.shape[1]
        Y   = image.shape[2]
        x1  = int(X/2) - int(self.max_pad_xy/2)
        x2  = int(X/2) + int(self.max_pad_xy/2)
        x1_ = max(x1, 0)
        x2_ = min(x2, X)
        y1  = int(Y/2) - int(self.max_pad_xy/2)
        y2  = int(Y/2) + int(self.max_pad_xy/2)
        y1_ = max(y1, 0)
        y2_ = min(y2, Y)
        z1  = int(Z/2) - int(self.max_pad_z /2)
        z2  = int(Z/2) + int(self.max_pad_z /2)
        z1_ = max(z1, 0)
        z2_ = min(z2, Z)
        image = image[z1_: z2_,x1_ : x2_,y1_ : y2_]
        image_pad = np.pad(image, ((z1_- z1, z2 - z2_),(x1_- x1, x2 - x2_), (y1_- y1, y2 - y2_)), 'constant')   
        return image_pad
    

    def padImg2D(self,image):
        X   = image.shape[0]
        Y   = image.shape[1]
        x1  = int(X/2) - int(self.max_pad_xy/2)
        x2  = int(X/2) + int(self.max_pad_xy/2)
        x1_ = max(x1, 0)
        x2_ = min(x2, X)
        y1  = int(Y/2) - int(self.max_pad_xy/2)
        y2  = int(Y/2) + int(self.max_pad_xy/2)
        y1_ = max(y1, 0)
        y2_ = min(y2, Y)
        image = image[x1_ : x2_,y1_ : y2_]
        image_pad = np.pad(image, ((x1_- x1, x2 - x2_), (y1_- y1, y2 - y2_)), 'constant')   
        return image_pad

    def pad_seq(self,data):
        return self.removeNaN(self.unitNormalisation(self.padImg(data)))
    
    def pad_img(self,data):
        return self.removeNaN(self.unitNormalisation(self.padImg2D(data)))
    
    def fix_labels(self,seg):
        seg[seg[:,:,:]==4]=3
        return seg

    def pad_seq_seg(self,data):
        return self.padImg(self.fix_labels(data))
    
    def getRestoreImg(self,data,orig_size):
        data   = self.padImg(data)
        Z      = orig_size.shape[0]
        X      = orig_size.shape[1]
        Y      = orig_size.shape[2]
        x1     = int(X/2) - int(self.max_pad_xy/2)
        x1_    = max(x1, 0)  
        y1     = int(Y/2) - int(self.max_pad_xy/2)
        y1_    = max(y1, 0)
        z1     = int(Z/2) - int(self.max_pad_z/2)
        z1_    = max(z1, 0)
        repad  = data[z1_-z1:z1_-z1 + Z,x1_-x1:x1_-x1 + X,y1_-y1:y1_-y1 + Y]
        return repad

    def seg(self,seg,save_path):
        for i in range(seg.shape[0]):
            seg[i][seg[i][:,:]==0] = 0
            seg[i][seg[i][:,:]==1] = 80
            seg[i][seg[i][:,:]==2] = 135
            seg[i][seg[i][:,:]==3] = 230
            cv2.imwrite(os.path.join(save_path,"seg_"+str(i)+".png"),seg[i])

    def img(self,img,save_path):
        for i in range(img.shape[0]):
            cv2.imwrite(os.path.join(save_path,"img_"+str(i)+".png"),255*img[i])

    def save_nifti_orig(self,open_path,save_path):
        # get stik of data
        open_stik              = sitk.ReadImage(open_path)
        # write information
        sitk.WriteImage(open_stik,save_path)
    

    def save_nifti_seg(self,open_path,save_path):
        # get stik of data
        open_stik              = sitk.ReadImage(open_path)
        # write information
        sitk.WriteImage(open_stik,save_path)
    
    def CopyInfo(self,gt,seg):
        gt_img       = sitk.GetArrayFromImage(gt)
        seg_img      = sitk.GetArrayFromImage(seg)
        collect_data = []
        for i in range(gt_img.shape[0]):
            collect_data.append(seg_img[i])
        seg_data = np.asarray(collect_data)
        seg_stik = sitk.GetImageFromArray(seg_data)
        seg_stik.CopyInformation(gt)
        return seg_stik 
    
    def fixSizeAccordingTemp(self,data,path_nii_tmp,flag):
        tmp     = sitk.GetArrayFromImage(sitk.ReadImage(path_nii_tmp)) 
        np_final = None
        if tmp.shape[0] > data.shape[0]:
            a_data  = []
            a_tmp   = []
            for i in range(data.shape[0]):
                a_data.append(data[i])
            for _ in range(tmp.shape[0]-data.shape[0]):
                a_tmp.append(np.zeros((tmp.shape[1],tmp.shape[2])))
            final = a_data + a_tmp
            np_final = np.asarray(final) 
        else:
            np_final = data
        if tmp.shape[1] > data.shape[1]:
            new_xy_size = []
            for i in range(np_final.shape[0]):
                if flag:
                    nnp_final_tmp = cv2.resize(np_final[i], (tmp.shape[2],tmp.shape[1]),interpolation=cv2.INTER_NEAREST) 
                    new_xy_size.append(nnp_final_tmp)
                else:
                    nnp_final_tmp = cv2.resize(np_final[i], (tmp.shape[2],tmp.shape[1]),interpolation=cv2.INTER_CUBIC) 
                    new_xy_size.append(nnp_final_tmp)
            np_final = np.asarray(new_xy_size) 
        return np_final

    def save_nifti_pred(self,data,save_path,path_nii_tmp,ctype,flag):
        data_new  = self.fixSizeAccordingTemp(data,path_nii_tmp,flag)
        tmp     = sitk.GetArrayFromImage(sitk.ReadImage(path_nii_tmp))
        # open tmp 
        get_stik_tmp    = sitk.ReadImage(path_nii_tmp)
        # get stik of data
        get_stik_input  = sitk.GetImageFromArray(data_new)
        get_stik_input  = sitk.Cast(get_stik_input, ctype)
        # copy information 
        #get_stik_input = self.CopyInfo(get_stik_input,get_stik_tmp)
        get_stik_input.CopyInformation(get_stik_tmp)
        # write information
        sitk.WriteImage(get_stik_input,save_path)
    
    '''
    def convert_uint8(self,img):
        # img is a numpy array/cv2 image
        img = img - img.min() # Now between 0 and 8674
        img = img / img.max() * 255
        return np.uint8(img)

    def getCLAHE(self,data):  
        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
        result_app = []
        kernel     = np.ones((3,3),np.float32)/9
        for z in range(data.shape[0]):
            clahe_img = clahe.apply(self.convert_uint8(data[z]))
            filter2d  = cv2.filter2D(clahe_img,-1,kernel).astype(np.float64)
            #blur      = cv2.blur(filter2d,(3,3)).astype(np.float64)
            filter2d = filter2d - filter2d.min() 
            filter2d = filter2d / filter2d.max() 
            result_app.append(filter2d)
        return np.asarray(result_app)
    '''

def test_seg():
    data_path     = "/home/bionick87/Desktop/14AA01658"
    save_path     = "/home/bionick87/Desktop/results_now"
    pre           = preprocessing(400,100)
    # Open sequence
    PATH_original_SEG_ED = os.path.join(data_path,"low_gray_genscan_ED.nii.gz")
    PATH_original_SEG_ES = os.path.join(data_path,"low_gray_genscan_ED.nii.gz")
    
    seg_ED_sitk   = sitk.ReadImage(PATH_original_SEG_ED)
    seg_ES_sitk   = sitk.ReadImage(PATH_original_SEG_ES)
    seg_ED        = sitk.GetArrayFromImage(seg_ED_sitk)
    seg_ES        = sitk.GetArrayFromImage(seg_ES_sitk)
    
    # Padding ED, ES segmentation at max size 

    seg_ED_pad    = pre.padImg(seg_ED)
    seg_ES_pad    = pre.padImg(seg_ES)

    clahe_data    = pre.getCLAHE(seg_ED_pad)

    print(clahe_data.shape)

    pre.img(clahe_data,save_path)


if __name__ == "__main__":
    test_seg()

    

