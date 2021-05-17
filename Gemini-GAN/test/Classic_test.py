##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################

import numpy as np 
from   tqdm import tqdm
import cv2,os
import SimpleITK  as sitk
import nibabel as nib
import shutil
from skimage.measure import compare_psnr, compare_ssim
import math

###############################################################################
# Before starting the code please configure the paths at the end of the code  #
###############################################################################


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    ssim_val = compare_ssim(gt, pred, multichannel=False, data_range=gt.max())
    if  math.isnan(ssim_val) or \
        ssim_val == -np.inf  or \
        ssim_val == np.inf:
            return 0
    else:
            return ssim_val

def removeNaN(listd):
    cleanedList = [x for x in listd if str(x) != 'nan']
    return cleanedList

def getPSNR(gt,pred):
    psnr_list = []
    for i in range(gt.shape[0]):
        val = psnr(gt[i],pred[i])
        # possibile error check i.e psnr == 0 is <= to black image
        if  math.isnan(val) or val == -np.inf  or val == np.inf or val == 0 or val<0:
            continue
        else:
            psnr_list.append(val)
    return np.mean(psnr_list)


def open_txt(path):
    data = []
    with open(path, 'r') as f:
        data = [line.strip() for line in f]
    return data

def save_gt(path_input_SR,path_input_LR,fr,typemode,path_save):
    nim_input_SR_open    = nib.load(path_input_SR)
    nim_input_LR_open    = nib.load(path_input_LR)
    path_save_sr_input   = os.path.join(path_save,"SR_"+typemode+"_INPUT_"+fr+".nii.gz")
    path_save_lr_input   = os.path.join(path_save,"LR_"+typemode+"_INPUT_"+fr+".nii.gz")
    nib.save(nim_input_SR_open, path_save_sr_input)
    nib.save(nim_input_LR_open, path_save_lr_input)

def save_predictions(path_input_SR,fr,typemode,path_save,prediction):
    nim_input_SR_open  = nib.load(path_input_SR)
    prediction         = np.transpose(prediction, (2, 1, 0))
    nim_pred_SR_open   = nib.Nifti1Image(prediction, nim_input_SR_open.affine, nim_input_SR_open.header)
    path_save_sr_pred  = os.path.join(path_save,"SR_"+typemode+"_PREDICTION_"+fr+".nii.gz")
    nib.save(nim_pred_SR_open,  path_save_sr_pred)

def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    os.system("chmod -R 777 " + file_path)

def readVol(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def fixSeg(seg):
    seg[seg[:,:,:]==4]=3
    return seg

def padImg(image,max_pad_xy=400,max_pad_z=100):
    Z   = image.shape[0]
    X   = image.shape[1]
    Y   = image.shape[2]
    x1  = int(X/2) - int(max_pad_xy/2)
    x2  = int(X/2) + int(max_pad_xy/2)
    x1_ = max(x1, 0)
    x2_ = min(x2, X)
    y1  = int(Y/2) - int(max_pad_xy/2)
    y2  = int(Y/2) + int(max_pad_xy/2)
    y1_ = max(y1, 0)
    y2_ = min(y2, Y)
    z1  = int(Z/2) - int(max_pad_z /2)
    z2  = int(Z/2) + int(max_pad_z /2)
    z1_ = max(z1, 0)
    z2_ = min(z2, Z)
    image = image[z1_: z2_,x1_ : x2_,y1_ : y2_]
    image_pad = np.pad(image, ((z1_- z1, z2 - z2_),(x1_- x1, x2 - x2_), (y1_- y1, y2 - y2_)), 'constant')   
    return image_pad

def getRestoreImg(data,orig_size,max_pad_xy=400,max_pad_z=100):
    data   = padImg(data)
    Z      = orig_size.shape[0]
    X      = orig_size.shape[1]
    Y      = orig_size.shape[2]
    x1     = int(X/2) - int(max_pad_xy/2)
    x1_    = max(x1, 0)  
    y1     = int(Y/2) - int(max_pad_xy/2)
    y1_    = max(y1, 0)
    z1     = int(Z/2) - int(max_pad_z/2)
    z1_    = max(z1, 0)
    repad  = data[z1_-z1:z1_-z1 + Z,x1_-x1:x1_-x1 + X,y1_-y1:y1_-y1 + Y]
    return repad

#https://mirtk.github.io/commands/resample-image.html

def resample_bspline(input_lr,output_lr,type): 
    if type:
        os.system('resample ' 
                '{0} '
                '{1} '
                ' -bspline -size 2 2 2 -gaussian 2 -isotropic 0.1'
                .format(input_lr, output_lr))
    else:
        os.system('resample ' 
                '{0} '
                '{1} '
                ' -bspline -size 2 2 2 -gaussian 0.3 -isotropic  0.1'
                .format(input_lr, output_lr))

    os.system("chmod -R 777 " + output_lr)

#linear
def resample_Linear(input_lr,output_lr,type): 
    if type:
        os.system('resample ' 
                '{0} '
                '{1} '
                ' -linear -size 2 2 2 -gaussian 2 -isotropic 0.1'
                .format(input_lr, output_lr))
    else:
        os.system('resample ' 
                '{0} '
                '{1} '
                ' -linear -size 2 2 2 -gaussian 0.3 -isotropic  0.1'.format(input_lr, output_lr))

    os.system("chmod -R 777 " + output_lr)



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


def Resize(low_res_path, save_path,typeSample):
    """Resamples an image to given element spacing and output size."""
    out_spacing=(3.0, 3.0, 3.0)
    out_size = None
    pad_value = 0
    image = sitk.ReadImage(low_res_path)
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)
    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    
    if typeSample   == "BSpline":
        resample.SetInterpolator(sitk.sitkBSpline)
    elif typeSample == "Linear":
        resample.SetInterpolator(sitk.sitkLinear)
    elif typeSample == "NN":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    out = resample.Execute(sitk.Cast(image, sitk.sitkFloat32))
    sitk.WriteImage(out,save_path)

def overallDice(pathsave,path_valid,type_data_valid,pathCode,type_method):
    #######################
    patients            = open_txt(os.path.join(pathCode,"dataset_txt",type_data_valid+".txt"))
    #######################
    wrong_patient       = []
    ed_global_endo_mean = []
    ed_global_myo_mean  = []
    ed_global_rv_mean   = []
    ed_global_endo_std  = []
    ed_global_myo_std   = []
    ed_global_rv_std    = []
    es_global_endo_mean = []
    es_global_myo_mean  = []
    es_global_rv_mean   = []
    es_global_endo_std  = []
    es_global_myo_std   = []
    es_global_rv_std    = []
    #######################
    ed_mean_psnr        = []
    ed_std_psnr         = []
    es_mean_psnr        = []
    es_std_psnr         = []
    ed_mean_ssim        = []
    ed_std_ssim         = []
    es_mean_ssim        = []
    es_std_ssim         = []
    #######################
    os.path.join(pathCode,"dataset_txt",type_data_valid+".txt")
    pathsave_method = os.path.join(pathsave,type_method)
    mkdir_dir(pathsave_method)
    #######################
    for i in tqdm(range(len(patients))):
        ########################################################################################
        save_path = os.path.join(pathsave_method,patients[i])
        mkdir_dir(save_path)
        tmp_path = os.path.join(save_path,"tmp")
        mkdir_dir(tmp_path)
        ########################################################################################
        # IMG ED
        path_low_gray_input_ED   = os.path.join(path_valid,patients[i],"low_gray_genscan_ED.nii.gz")
        path_hight_gray_input_ED = os.path.join(path_valid,patients[i],"hight_gray_genscan_ED.nii.gz")
        low_gray_input_ED        = readVol(path_low_gray_input_ED)
        hight_gray_input_ED      = readVol(path_hight_gray_input_ED)
        # SEG ED
        path_low_seg_input_ED    = os.path.join(path_valid,patients[i],"low_seg_genscan_ED.nii.gz")
        path_hight_seg_input_ED  = os.path.join(path_valid,patients[i],"hight_seg_genscan_ED.nii.gz")
        low_seg_input_ED         = fixSeg(readVol(path_low_seg_input_ED))
        hight_seg_input_ED       = fixSeg(readVol(path_hight_seg_input_ED))
        ########################################################################################
        # IMG ES
        path_low_gray_input_ES   = os.path.join(path_valid,patients[i],"low_gray_genscan_ES.nii.gz")
        path_hight_gray_input_ES = os.path.join(path_valid,patients[i],"hight_gray_genscan_ES.nii.gz")
        low_gray_input_ES        = readVol(path_low_gray_input_ES)
        hight_gray_input_ES      = readVol(path_hight_gray_input_ES)
        # SEG ES
        path_low_seg_input_ES    = os.path.join(path_valid,patients[i],"low_seg_genscan_ES.nii.gz")
        path_hight_seg_input_ES  = os.path.join(path_valid,patients[i],"hight_seg_genscan_ES.nii.gz")
        low_seg_input_ES         = fixSeg(readVol(path_low_seg_input_ES))
        hight_seg_input_ES       = fixSeg(readVol(path_hight_seg_input_ES))
        ########################################################################################

        '''
        if type_method =="Linear":
            path_resample_low_gray_genscan_ED = os.path.join(tmp_path,"resample_low_gray_genscan_ED.nii.gz")
            path_resample_low_seg_input_ED = os.path.join(tmp_path,"resample_low_seg_genscan_ED.nii.gz")
            path_resample_low_gray_genscan_ES = os.path.join(tmp_path,"resample_low_gray_genscan_ES.nii.gz")
            path_resample_low_seg_input_ES = os.path.join(tmp_path,"resample_low_seg_genscan_ES.nii.gz")
            resample_Linear(path_low_gray_input_ED,path_resample_low_gray_genscan_ED,True)
            resample_Linear(path_low_seg_input_ED,path_resample_low_seg_input_ED,False)
            resample_Linear(path_low_gray_input_ES,path_resample_low_gray_genscan_ES,True)
            resample_Linear(path_low_seg_input_ES,path_resample_low_seg_input_ES,False)
        elif type_method =="Bspline":
            path_resample_low_gray_genscan_ED = os.path.join(tmp_path,"resample_low_gray_genscan_ED.nii.gz")
            path_resample_low_seg_input_ED = os.path.join(tmp_path,"resample_low_seg_genscan_ED.nii.gz")
            path_resample_low_gray_genscan_ES = os.path.join(tmp_path,"resample_low_gray_genscan_ES.nii.gz")
            path_resample_low_seg_input_ES = os.path.join(tmp_path,"resample_low_seg_genscan_ES.nii.gz")
            resample_bspline(path_low_seg_input_ED,path_resample_low_seg_input_ED,False)
            resample_bspline(path_low_seg_input_ES,path_resample_low_seg_input_ES,False)
            resample_bspline(path_low_gray_input_ED,path_resample_low_gray_genscan_ED,True)
            resample_bspline(path_low_gray_input_ES,path_resample_low_gray_genscan_ES,True)
        '''

        ######################################
        path_resample_low_gray_genscan_ED = os.path.join(tmp_path,"resample_low_gray_genscan_ED.nii.gz")
        path_resample_low_gray_genscan_ES = os.path.join(tmp_path,"resample_low_gray_genscan_ES.nii.gz")
        path_resample_low_seg_input_ED = os.path.join(tmp_path,"resample_low_seg_genscan_ED.nii.gz")
        path_resample_low_seg_input_ES = os.path.join(tmp_path,"resample_low_seg_genscan_ES.nii.gz")
        ######################################
        Resize(path_low_gray_input_ED,path_resample_low_gray_genscan_ED,type_method)
        Resize(path_low_gray_input_ES,path_resample_low_gray_genscan_ES,type_method)
        Resize(path_low_seg_input_ED,path_resample_low_seg_input_ED,type_method)
        Resize(path_low_seg_input_ES,path_resample_low_seg_input_ES,type_method)
        ######################################

        ######################################
        open_seg_ED = readVol(path_resample_low_seg_input_ED)
        open_seg_ES = readVol(path_resample_low_seg_input_ES)
        open_img_ED = readVol(path_resample_low_gray_genscan_ED)
        open_img_ES = readVol(path_resample_low_gray_genscan_ES)
        ######################################
        save_predictions(path_resample_low_gray_genscan_ED,"ED","GRAY",save_path,open_img_ED)
        save_predictions(path_resample_low_gray_genscan_ES,"ES","GRAY",save_path,open_img_ES)
        save_predictions(path_resample_low_seg_input_ED,"ED","SEG",save_path,open_seg_ED)
        save_predictions(path_resample_low_seg_input_ES,"ES","SEG",save_path,open_seg_ES)
        ######################################
        seg_ED    = getRestoreImg(open_seg_ED,hight_seg_input_ED)
        seg_ES    = getRestoreImg(open_seg_ES,hight_seg_input_ES)
        img_ED    = getRestoreImg(open_img_ED,hight_gray_input_ED)
        img_ES    = getRestoreImg(open_img_ES,hight_gray_input_ES)
        endo_ed,myo_ed,rv_ed =  DiceEval(seg_ED,hight_seg_input_ED)
        endo_es,myo_es,rv_es =  DiceEval(seg_ES,hight_seg_input_ES)
        ######################################
        save_gt(path_hight_gray_input_ED,path_low_gray_input_ED,"ED","GRAY",save_path)
        save_gt(path_hight_gray_input_ES,path_low_gray_input_ES,"ES","GRAY",save_path)
        save_gt(path_hight_seg_input_ED,path_low_seg_input_ED,"ED","SEG",save_path)
        save_gt(path_hight_seg_input_ES,path_low_seg_input_ES,"ES","SEG",save_path)
        ######################################
        ed_global_endo_mean.append(endo_ed[0])
        ed_global_endo_std.append(endo_ed[1])
        ed_global_myo_mean.append(myo_ed[0])
        ed_global_myo_std.append(myo_ed[1])
        ed_global_rv_mean.append(rv_ed[0])
        ed_global_rv_std.append(rv_ed[1])
        es_global_endo_mean.append(endo_es[0])
        es_global_endo_std.append(endo_es[1])
        es_global_myo_mean.append(myo_es[0])
        es_global_myo_std.append(myo_es[1])     
        es_global_rv_mean.append(rv_es[0])
        es_global_rv_std.append(rv_es[1])
        #####################################
        ED_psnr_gray = PSNR (hight_gray_input_ED,img_ED,hight_seg_input_ED)
        ES_psnr_gray = PSNR (hight_gray_input_ES,img_ES,hight_seg_input_ES)
        ED_ssim_gray = SSIM (hight_gray_input_ED,img_ED,hight_seg_input_ED)
        ES_ssim_gray = SSIM (hight_gray_input_ES,img_ES,hight_seg_input_ES)
        #####################################
        ed_mean_psnr.append(ED_psnr_gray[0])
        ed_std_psnr.append(ED_psnr_gray[1])
        es_mean_psnr.append(ES_psnr_gray[0])
        es_std_psnr.append(ES_psnr_gray[1])
        ed_mean_ssim.append(ED_ssim_gray[0])
        ed_std_ssim.append(ED_ssim_gray[1])
        es_mean_ssim.append(ES_ssim_gray[0])
        es_std_ssim.append(ES_ssim_gray[1])
        shutil.rmtree(tmp_path)
    ###########################################
    global_mean_endo_ed = np.mean(removeNaN(ed_global_endo_mean))
    global_std_endo_ed = np.std(removeNaN(ed_global_endo_std))
    global_mean_myo_ed = np.mean(removeNaN(ed_global_myo_mean))
    global_std_myo_ed = np.std(removeNaN(ed_global_myo_std))
    global_mean_rv_ed = np.mean(removeNaN(ed_global_rv_mean))
    global_std_rv_ed = np.std(removeNaN(ed_global_rv_std))
    ###########################################
    global_mean_endo_es = np.mean(removeNaN(es_global_endo_mean))
    global_std_endo_es = np.std(removeNaN(es_global_endo_std))
    global_mean_myo_es = np.mean(removeNaN(es_global_myo_mean))
    global_std_myo_es = np.std(removeNaN(es_global_myo_std))
    global_mean_rv_es = np.mean(removeNaN(es_global_rv_mean))
    global_std_rv_es = np.std(removeNaN(es_global_rv_std))
    ###########################################
    ed_mean_psnr_global = np.mean(removeNaN(ed_mean_psnr))
    ed_std_psnr_global  = np.std(removeNaN(ed_std_psnr))
    es_mean_psnr_global = np.mean(removeNaN(es_mean_psnr))
    es_std_psnr_global = np.std(removeNaN(es_std_psnr))
    ed_mean_ssim_global = np.mean(removeNaN(ed_mean_ssim))
    ed_std_ssim_global = np.std(removeNaN(ed_std_ssim))
    es_mean_ssim_global = np.mean(removeNaN(es_mean_ssim))
    es_std_ssim_global = np.std(removeNaN(es_std_ssim))
    ###########################################
    str_save = "... Total patients " + str(len(ed_mean_psnr))                                    + "\n" + \
               " ----------------------------------------- "                                     + "\n" + \
               " ... ENDO ED  "      + str(global_mean_endo_ed) + "/" + str(global_std_endo_ed)  + "\n" + \
               " ... MYO ED  "       + str(global_mean_myo_ed)  + "/" + str(global_std_myo_ed)   + "\n" + \
               " ... RV ED  "        + str(global_mean_rv_ed)   + "/" + str(global_std_rv_ed)    + "\n" + \
               " ----------------------------------------- "                                     + "\n" + \
               " ... ENDO ES  "      + str(global_mean_endo_es) + "/" + str(global_std_endo_es)  + "\n" + \
               " ... MYO ES  "       + str(global_mean_myo_es)  + "/" + str(global_std_myo_es)   + "\n" + \
               " ... RV ES  "        + str(global_mean_rv_es)   + "/" + str(global_std_rv_es)    + "\n" + \
               " ----------------------------------------- "                                     + "\n" + \
               " ... PSNR ED  "      + str(ed_mean_psnr_global) + "/" + str(ed_std_psnr_global)  + "\n" + \
               " ... SSIM ED  "      + str(ed_mean_ssim_global) + "/" + str(ed_std_ssim_global)  + "\n" + \
               " ----------------------------------------- "                                     + "\n" + \
               " ... PSNR ES  "      + str(es_mean_psnr_global) + "/" + str(es_std_psnr_global)  + "\n" + \
               " ... SSIM ES  "      + str(es_mean_ssim_global) + "/" + str(es_std_ssim_global)
    text_file = open(os.path.join(pathsave_method,type_method + "_table.txt"), "w")
    text_file.write(str_save)
    text_file.close()


if __name__ == "__main__":
   
    ############################
    path_code       = "" # Folder where is the list of patients (named dataset_txt)
    path_valid      = "" # Folder where is LR/HR dataset 
    pathsave        = "" # Folder where to save  
    ############################
    type_data_valid = "test" # Type of dataset 
    type_method     = "BSpline" # Please select Linear/NN/BSpline
    ############################
    overallDice(pathsave,path_valid,type_data_valid,path_code,type_method)
    ############################
