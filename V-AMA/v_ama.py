##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################

import torch
import torch.nn as nn
from   torch.autograd import Variable
import torch.optim as optim
import torchvision
from   tqdm import tqdm
import model
from   util import *
import os
import itertools
import numpy as np 
from   torch.utils.data import DataLoader
from   dataset import TrainDataset
from   datetime import date
from   preprocessing import preprocessing
import SimpleITK  as sitk
from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import structural_similarity as ssim
import math
from plots import plots
import cv2
import copy
import warnings
from pretrain import *
from torchvision import transforms, utils
warnings.simplefilter("ignore",  category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 


###############################################################################
# Before starting the code please configure the paths at the end of the code  #
###############################################################################


def Resize(path_lr,path_sr):
    image_LR           = sitk.ReadImage(path_lr)
    image_SR           = sitk.ReadImage(path_sr)
    out_spacing        = np.array(image_LR.GetSpacing())
    original_spacing   = np.array(image_LR.GetSpacing())
    original_size      = np.array(image_LR.GetSize())
    out_size           = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    original_direction = np.array(image_SR.GetDirection()).reshape(len(original_spacing),-1)
    original_center    = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center         = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    original_center    = np.matmul(original_direction, original_center)
    out_center         = np.matmul(original_direction, out_center)
    out_origin         = np.array(image_LR.GetOrigin()) + (original_center - out_center)
    resample           = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image_LR.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    out = resample.Execute(sitk.Cast(image_SR, sitk.sitkUInt8))
    return sitk.GetArrayFromImage(out)


def unitNormalisation(v):        
    v_min = np.amin(v)
    v_max = np.amax(v)
    out   = (v - v_min)/(v_max - v_min)
    return out  

def brightness(img):
    color_jitter = transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5)
    transform = transforms.Compose([transforms.ColorJitter.get_params(
    color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
    color_jitter.hue),transforms.ToTensor()])
    list_data = []
    for z in range(img.shape[0]):
        data = transform(transforms.ToPILImage()(torch.from_numpy(img[z,:,:]).float()))[0]
        list_data.append(data)
    np_array = torch.stack(list_data).numpy()
    return  unitNormalisation(np_array)

def brightnessImg(img):
    color_jitter = transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5)
    transform = transforms.Compose([transforms.ColorJitter.get_params(
    color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
    color_jitter.hue),transforms.ToTensor()])
    data = transform(transforms.ToPILImage()(torch.from_numpy(img).float()))[0].numpy()
    return  unitNormalisation(data)

def removeNaN(listd):
    cleanedList = [x for x in listd if str(x) != 'nan']
    return cleanedList


def load(path_model):
    generator     = model.Open(100,100)
    generator.load_state_dict(torch.load(path_model))
    return generator.cuda().eval().float()

def open_txt(path):
    data = []
    with open(path, 'r') as f:
        data = [line.strip() for line in f]
    return data

def get_torch(data):
    torch_out = torch.from_numpy(data).unsqueeze(0).to("cuda")
    torch_out[torch_out != torch_out] = 0
    return Variable(torch_out)

def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

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
        

def DiceEvalDA(seg,gt):
    seg = fix_labels(seg)
    gt  = fix_labels(gt)
    weights = []
    mean = 0
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
            weights.append((endo+myo+rv)/3)
    if weights == []:
       return 0
    else:
        return np.mean(removeNaN(weights))

def SingleInference3D(Es,Et,Gs,Gt,Ns,Nt,input_img,typedata):
    pre = preprocessing(400,100)
    output_img_cuda = None
    with torch.no_grad():
        input_img_cuda = Variable(get_torch(input_img).to("cuda").float())
        
        if typedata == "MUNIT":
            a1,b1      = Es(input_img_cuda)
            c_code_1   = a1[0]
            cf_1       = a1[1]
            s_code_1   = b1
            output_img_cuda = Gs(c_code_1, input_img_cuda, cf_1, s_code_1)

        elif typedata == "UNIT":
            c_code, s_code, f_code  = Es(input_img_cuda)
            output_img_cuda = Gs(s_code,input_img_cuda,f_code)
        
        elif typedata == "CycleGAN":
            output_img_cuda = Es(input_img_cuda,False)    
        
        elif typedata == "BicycleGAN": 
            random_z = var(torch.randn(1,8))
            output_img_cuda = Es(input_img_cuda, random_z,False)  

        elif typedata == "V_AMA": 
            et,ft            = Et(input_img_cuda)
            es,fs            = Es(input_img_cuda)
            mu_t, log_var_t  = Nt(et)
            mu_s, log_var_s  = Ns(es)
            std_t            = torch.exp(log_var_t / 2)
            std_s            = torch.exp(log_var_s / 2)
            random_z_t       = var(torch.randn(1, 320000))
            random_z_t       = random_z_t*(std_s+std_t) + (mu_s+mu_t)
            random_z_t       = random_z_t.view(es.size())
            fm               = [fs[0]+ft[0].detach(),fs[1]+ft[1].detach(),fs[2]+ft[2].detach(),fs[3]+ft[3].detach()]
            output_img_cuda  = Gs(random_z_t,input_img_cuda,fm)

    return output_img_cuda.cpu().numpy()[0]


def validFunction(Es,Et,Gs,Gt,Ns,Nt,jointGenerator,pathCode,pathData,pathSave,typedata,typemodel,flag_noise):   
    Es.eval()
    Et.eval()
    Gs.eval()
    Gt.eval()
    Ns.eval()
    Nt.eval()
    jointGenerator.eval()
    #######################
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
    ed_mean_psnr = []
    ed_std_psnr  = []
    es_mean_psnr = []
    es_std_psnr  = []
    ed_mean_ssim = []
    ed_std_ssim  = []
    es_mean_ssim = []
    es_std_ssim  = []
    #######################

    patients  = open_txt(os.path.join(pathCode,"dataset_txt","DA","valid.txt"))
    pre = preprocessing(400,100)
    print("\n\n\n  ... Test")
    for i in tqdm(range(len(patients))):
        with torch.no_grad():
            
            patient = patients[i]
            save_path = os.path.join(pathSave,patient)
            mkdir_dir(save_path)
            path_low_gray_input_ED   = None
            path_low_gray_input_ES   = None
            path_hight_gray_input_ED = None
            path_hight_gray_input_ES = None
            path_hight_seg_input_ED  = None
            path_hight_seg_input_ES  = None
            pad_target_img_ed_low    = None
            pad_target_img_es_low    = None
            
            path_low_gray_input_ED = os.path.join(pathData,patient,"low_gray_SG_ED.nii.gz")
            path_low_gray_input_ES = os.path.join(pathData,patient,"low_gray_SG_ES.nii.gz")

            path_hight_gray_input_ED = os.path.join(pathData,patient,"hight_gray_SG_ED.nii.gz")
            path_hight_gray_input_ES = os.path.join(pathData,patient,"hight_gray_SG_ES.nii.gz")

            path_hight_seg_input_ED = os.path.join(pathData,patient,"hight_seg_SG_ED.nii.gz")
            path_hight_seg_input_ES = os.path.join(pathData,patient,"hight_seg_SG_ES.nii.gz")

            input_img_ed_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ED))
            input_img_es_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ES))

            pad_input_img_ed_low = pre.pad_seq(input_img_ed_low)
            pad_input_img_es_low = pre.pad_seq(input_img_es_low)
            
            if flag_noise:  

                input_img_ed_low_br = brightness(pad_input_img_ed_low)
                input_img_ed_low_br_th = torch.from_numpy(input_img_ed_low_br).float()
                scanner_noise = torch.FloatTensor(input_img_ed_low_br_th.size()).normal_(0,1)
                nosie_input_img_ed_low = torch.add(input_img_ed_low_br_th,0.05*scanner_noise)
                input_img_es_low_br = brightness(pad_input_img_es_low)
                input_img_es_low_br_th = torch.from_numpy(input_img_es_low_br).float()
                scanner_noise = torch.FloatTensor(input_img_es_low_br_th.size()).normal_(0,1)
                nosie_input_img_es_low = torch.add(input_img_es_low_br_th,0.05*scanner_noise)
                nosie_input_img_es_low[nosie_input_img_es_low != nosie_input_img_es_low] = 0
                path_low_gray_input_ED = os.path.join(pathData,patient,"low_gray_SG_ED.nii.gz")
                path_low_gray_input_ES = os.path.join(pathData,patient,"low_gray_SG_ES.nii.gz")
                path_low_input_noise_ED = os.path.join(save_path,"low_input_noise_ED.nii.gz")
                path_low_input_noise_ES = os.path.join(save_path,"low_input_noise_ES.nii.gz")
                ed_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ED))
                es_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ES))
                cpu_ed_t = pre.getRestoreImg(nosie_input_img_ed_low.numpy(),ed_gt_low)
                cpu_es_t = pre.getRestoreImg(nosie_input_img_es_low.numpy(),es_gt_low)
                pre.save_nifti_pred(cpu_ed_t,path_low_input_noise_ED,path_low_gray_input_ED,sitk.sitkFloat64,False)
                pre.save_nifti_pred(cpu_es_t,path_low_input_noise_ES,path_low_gray_input_ES,sitk.sitkFloat64,False)

                norm_nosie_input_img_ed_low = pre.pad_seq(nosie_input_img_ed_low.numpy())
                norm_nosie_input_img_es_low = pre.pad_seq(nosie_input_img_es_low.numpy())

                target_img_ed_low = SingleInference3D(Es,Et,Gs,Gt,Ns,Nt,norm_nosie_input_img_ed_low,typemodel)
                target_img_es_low = SingleInference3D(Es,Et,Gs,Gt,Ns,Nt,norm_nosie_input_img_es_low,typemodel)

                pad_target_img_ed_low = pre.pad_seq(target_img_ed_low)
                pad_target_img_es_low = pre.pad_seq(target_img_es_low)
            
            else:
                target_img_ed_low = SingleInference3D(Es,Et,Gs,Gt,Ns,Nt,pad_input_img_ed_low,typemodel)
                target_img_es_low = SingleInference3D(Es,Et,Gs,Gt,Ns,Nt,pad_input_img_es_low,typemodel)
                 
                path_low_gray_input_ED = os.path.join(pathData,patient,"low_gray_ukbb_ED.nii.gz")
                path_low_gray_input_ES = os.path.join(pathData,patient,"low_gray_ukbb_ES.nii.gz")

                path_low_input_out_ED = os.path.join(save_path,"low_input_da_ED.nii.gz")
                path_low_input_out_ES = os.path.join(save_path,"low_input_da_ES.nii.gz")

                ed_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ED))
                es_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ES))

                cpu_ed = pre.getRestoreImg(target_img_ed_low,ed_gt_low)
                cpu_es = pre.getRestoreImg(target_img_es_low,es_gt_low)
                
                pre.save_nifti_pred(cpu_ed,path_low_input_out_ED,path_low_gray_input_ED,sitk.sitkFloat64,False)
                pre.save_nifti_pred(cpu_es,path_low_input_out_ES,path_low_gray_input_ES,sitk.sitkFloat64,False)

                open_input_img_ed_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_input_out_ED))
                open_input_img_es_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_input_out_ES))
                
                pad_target_img_ed_low = pre.pad_seq(open_input_img_ed_low)
                pad_target_img_es_low = pre.pad_seq(open_input_img_es_low)

            cuda_img_ed_target_cuda = Variable(get_torch(pad_target_img_ed_low).to("cuda").float())
            cuda_img_es_target_cuda = Variable(get_torch(pad_target_img_es_low).to("cuda").float())

            hight_ed_pred_img_target,\
            hight_ed_pred_seg_target = jointGenerator(cuda_img_ed_target_cuda)

            hight_es_pred_img_target,\
            hight_es_pred_seg_target = jointGenerator(cuda_img_es_target_cuda)

            _, hight_ed_pred_seg_target  = torch.max (hight_ed_pred_seg_target, dim=1)
            cpu_hight_ed_pred_seg_target = hight_ed_pred_seg_target.cpu().data.numpy()

            _, hight_es_pred_seg_target  = torch.max (hight_es_pred_seg_target, dim=1)
            cpu_hight_es_pred_seg_target = hight_es_pred_seg_target.cpu().data.numpy()

            cpu_hight_ed_pred_img_target = hight_ed_pred_img_target.cpu().data.numpy()
            cpu_hight_es_pred_img_target = hight_es_pred_img_target.cpu().data.numpy()
                
            ####################################################################
            gt_seg_ed_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ED))
            gt_seg_es_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ES))
            ####################################################################

            ####################################################################
            gt_img_ed_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_gray_input_ED))
            gt_img_es_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_gray_input_ES))
            ####################################################################

            ####################################################################
            hight_ed_pred_target_seg = pre.getRestoreImg(cpu_hight_ed_pred_seg_target[0],gt_img_ed_hight)
            hight_es_pred_target_seg = pre.getRestoreImg(cpu_hight_es_pred_seg_target[0],gt_img_es_hight)
            ####################################################################

            ####################################################################
            hight_ed_pred_target_img = pre.getRestoreImg(cpu_hight_ed_pred_img_target[0],gt_seg_ed_hight)
            hight_es_pred_target_img = pre.getRestoreImg(cpu_hight_es_pred_img_target[0],gt_seg_es_hight)
            ####################################################################
            
            path_gt_SG_seg_ED = os.path.join(save_path,"gt_SR_seg_SG_ED.nii.gz")
            path_gt_SG_seg_ES = os.path.join(save_path,"gt_SR_seg_SG_ES.nii.gz")

            path_gt_SG_img_ED = os.path.join(save_path,"gt_SR_img_SG_ED.nii.gz")
            path_gt_SG_img_ES = os.path.join(save_path,"gt_SR_img_SG_ES.nii.gz")

            path_pred_SG_seg_ED = os.path.join(save_path,"pred_SR_seg_SG_ED.nii.gz")
            path_pred_SG_seg_ES = os.path.join(save_path,"pred_SR_seg_SG_ES.nii.gz")

            path_pred_SG_img_ED = os.path.join(save_path,"pred_SR_img_SG_ED.nii.gz")
            path_pred_SG_img_ES = os.path.join(save_path,"pred_SR_img_SG_ES.nii.gz")
            
            path_pred_SG_lr_ED  = os.path.join(save_path,"pred_DA_LR_ED.nii.gz")
            path_pred_SG_lr_ES  = os.path.join(save_path,"pred_DA_LR_ES.nii.gz")
            ####################################################################

            ####################################################################
            pre.save_nifti_pred(gt_seg_ed_hight,path_gt_SG_seg_ED,path_hight_seg_input_ED,sitk.sitkFloat64,True)
            pre.save_nifti_pred(gt_seg_es_hight,path_gt_SG_seg_ES,path_hight_seg_input_ES,sitk.sitkFloat64,True)
            ####################################################################

            ####################################################################
            pre.save_nifti_pred(hight_ed_pred_target_seg,path_pred_SG_seg_ED,path_hight_seg_input_ED,sitk.sitkFloat64,True)
            pre.save_nifti_pred(hight_es_pred_target_seg,path_pred_SG_seg_ES,path_hight_seg_input_ES,sitk.sitkFloat64,True)
            ####################################################################

            ####################################################################
            pre.save_nifti_pred(gt_img_ed_hight,path_gt_SG_img_ED,path_hight_gray_input_ED,sitk.sitkFloat64,False)
            pre.save_nifti_pred(gt_img_es_hight,path_gt_SG_img_ES,path_hight_gray_input_ES,sitk.sitkFloat64,False)
            ####################################################################
            
            ####################################################################
            pre.save_nifti_pred(hight_ed_pred_target_img,path_pred_SG_img_ED,path_hight_gray_input_ED,sitk.sitkFloat64,False)
            pre.save_nifti_pred(hight_es_pred_target_img,path_pred_SG_img_ES,path_hight_gray_input_ES,sitk.sitkFloat64,False)
            ####################################################################
                        
            ####################################################################
            endo_ed,myo_ed,rv_ed =  DiceEval(hight_ed_pred_target_seg,gt_seg_ed_hight)
            endo_es,myo_es,rv_es =  DiceEval(hight_es_pred_target_seg,gt_seg_es_hight)
            ####################################################################

            ######################################
            ed_global_endo_mean.append(endo_ed[0])
            ed_global_endo_std.append(endo_ed[1])
            ######################################
            ed_global_myo_mean.append(myo_ed[0])
            ed_global_myo_std.append(myo_ed[1])
            ######################################
            ed_global_rv_mean.append(rv_ed[0])
            ed_global_rv_std.append(rv_ed[1])
            ######################################
            es_global_endo_mean.append(endo_es[0])
            es_global_endo_std.append(endo_es[1])
            ######################################
            es_global_myo_mean.append(myo_es[0])
            es_global_myo_std.append(myo_es[1])   
            ######################################  
            es_global_rv_mean.append(rv_es[0])
            es_global_rv_std.append(rv_es[1])
            #####################################


            ED_psnr_gray = PSNR (sitk.GetArrayFromImage(sitk.ReadImage(path_hight_gray_input_ED)),\
                                 sitk.GetArrayFromImage(sitk.ReadImage(path_pred_SG_img_ED)),\
                                 sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ED)))

            ES_psnr_gray = PSNR (sitk.GetArrayFromImage(sitk.ReadImage(path_hight_gray_input_ES)),\
                                sitk.GetArrayFromImage(sitk.ReadImage(path_pred_SG_img_ES)),\
                                sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ES)))

            ED_ssim_gray = SSIM (sitk.GetArrayFromImage(sitk.ReadImage(path_hight_gray_input_ED)),\
                                 sitk.GetArrayFromImage(sitk.ReadImage(path_pred_SG_img_ED)),\
                                 sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ED)))

            ES_ssim_gray = SSIM (sitk.GetArrayFromImage(sitk.ReadImage(path_hight_gray_input_ES)),\
                                sitk.GetArrayFromImage(sitk.ReadImage(path_pred_SG_img_ES)),\
                                sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ES)))

            #####################################
            ed_mean_psnr.append(ED_psnr_gray[0])
            ed_std_psnr.append(ED_psnr_gray[1])

            es_mean_psnr.append(ES_psnr_gray[0])
            es_std_psnr.append(ES_psnr_gray[1])
            
            ed_mean_ssim.append(ED_ssim_gray[0])
            ed_std_ssim.append(ED_ssim_gray[1])
            
            es_mean_ssim.append(ES_ssim_gray[0])
            es_std_ssim.append(ES_ssim_gray[1])
            ####################################

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
    ###
    es_mean_psnr_global = np.mean(removeNaN(es_mean_psnr))
    es_std_psnr_global  = np.std(removeNaN(es_std_psnr))
    ###
    ed_mean_ssim_global = np.mean(removeNaN(ed_mean_ssim))
    ed_std_ssim_global  = np.std(removeNaN(ed_std_ssim))
    ###
    es_mean_ssim_global = np.mean(removeNaN(es_mean_ssim))
    es_std_ssim_global  = np.std(removeNaN(es_std_ssim))
    ###
    mean_dice_ed = (global_mean_endo_ed + global_mean_myo_ed + global_mean_rv_ed)/3
    mean_dice_es = (global_mean_endo_es + global_mean_myo_es + global_mean_rv_es)/3
    mean_psnr_ed = ed_mean_psnr_global
    mean_psnr_es = es_mean_psnr_global
    mean_ssim_ed = ed_mean_ssim_global
    mean_ssim_es = es_mean_ssim_global

    return [[mean_dice_ed,mean_dice_es],\
            [mean_psnr_ed,mean_psnr_es],\
            [mean_ssim_ed,mean_ssim_es]] 



def validFunctionUKBB(Es,Et,Gs,Gt,Ns,Nt,generator_retrain,pathCode,pathData,pathSave,typedata,typemodel,flag_noise):   
    Es.eval()
    Et.eval()
    Gs.eval()
    Gt.eval()
    Ns.eval()
    Nt.eval()
    generator_retrain.eval()
    #######################
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
    ed_mean_psnr = []
    ed_std_psnr  = []
    es_mean_psnr = []
    es_std_psnr  = []
    ed_mean_ssim = []
    ed_std_ssim  = []
    es_mean_ssim = []
    es_std_ssim  = []
    #######################
    patients  = open_txt(os.path.join(pathCode,"dataset_txt","valid.txt"))
    pre = preprocessing(400,100)
    print("\n\n\n  ... Test")
    for i in tqdm(range(len(patients))):
        with torch.no_grad():
            patient = patients[i]
            save_path = os.path.join(pathSave,patient)
            mkdir_dir(save_path)
            path_low_gray_input_ED   = None
            path_low_gray_input_ES   = None
            path_hight_seg_input_ED  = None
            path_hight_seg_input_ES  = None
            pad_target_img_ed_low    = None
            pad_target_img_es_low    = None
            
            path_low_gray_input_ED = os.path.join(pathData,patient,"low_gray_ukbb_ED.nii.gz")
            path_low_gray_input_ES = os.path.join(pathData,patient,"low_gray_ukbb_ES.nii.gz")

            path_hight_seg_input_ED = os.path.join(pathData,patient,"hight_seg_ukbb_ED.nii.gz")
            path_hight_seg_input_ES = os.path.join(pathData,patient,"hight_seg_ukbb_ES.nii.gz")

            path_low_seg_input_ED = os.path.join(pathData,patient,"low_seg_ukbb_ED.nii.gz")
            path_low_seg_input_ES = os.path.join(pathData,patient,"low_seg_ukbb_ES.nii.gz")

            input_img_ed_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ED))
            input_img_es_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ES))

            pad_input_img_ed_low = pre.pad_seq(input_img_ed_low)
            pad_input_img_es_low = pre.pad_seq(input_img_es_low)

            target_img_ed_low = SingleInference3D(Es,Et,Gs,Gt,Ns,Nt,pad_input_img_ed_low,typemodel)
            target_img_es_low = SingleInference3D(Es,Et,Gs,Gt,Ns,Nt,pad_input_img_es_low,typemodel)
                
            path_low_gray_input_ED = os.path.join(pathData,patient,"low_gray_ukbb_ED.nii.gz")
            path_low_gray_input_ES = os.path.join(pathData,patient,"low_gray_ukbb_ES.nii.gz")

            path_low_input_out_ED = os.path.join(save_path,"low_input_da_ukbb_ED.nii.gz")
            path_low_input_out_ES = os.path.join(save_path,"low_input_da_ukbb_ES.nii.gz")

            ed_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ED))
            es_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ES))

            cpu_ed = pre.getRestoreImg(target_img_ed_low,ed_gt_low)
            cpu_es = pre.getRestoreImg(target_img_es_low,es_gt_low)
            
            pre.save_nifti_pred(cpu_ed,path_low_input_out_ED,path_low_gray_input_ED,sitk.sitkFloat32,False)
            pre.save_nifti_pred(cpu_es,path_low_input_out_ES,path_low_gray_input_ES,sitk.sitkFloat32,False)

            open_input_img_ed_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_input_out_ED))
            open_input_img_es_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_input_out_ES))
            
            pad_target_img_ed_low = pre.pad_seq(open_input_img_ed_low)
            pad_target_img_es_low = pre.pad_seq(open_input_img_es_low)

            cuda_img_ed_target_cuda = Variable(get_torch(pad_target_img_ed_low).to("cuda").float())
            cuda_img_es_target_cuda = Variable(get_torch(pad_target_img_es_low).to("cuda").float())

            hight_ed_pred_img_target,\
            hight_ed_pred_seg_target = generator_retrain(cuda_img_ed_target_cuda)

            hight_es_pred_img_target,\
            hight_es_pred_seg_target = generator_retrain(cuda_img_es_target_cuda)

            _, hight_ed_pred_seg_target  = torch.max (hight_ed_pred_seg_target, dim=1)
            cpu_hight_ed_pred_seg_target = hight_ed_pred_seg_target.cpu().data.numpy()

            _, hight_es_pred_seg_target  = torch.max (hight_es_pred_seg_target, dim=1)
            cpu_hight_es_pred_seg_target = hight_es_pred_seg_target.cpu().data.numpy()

            cpu_hight_ed_pred_img_target = hight_ed_pred_img_target.cpu().data.numpy()
            cpu_hight_es_pred_img_target = hight_es_pred_img_target.cpu().data.numpy()
                
            ####################################################################
            gt_seg_ed_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ED))
            gt_seg_es_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ES))
            ####################################################################

            ####################################################################
            gt_seg_ed_low = fix_labels(sitk.GetArrayFromImage(sitk.ReadImage(path_low_seg_input_ED)))
            gt_seg_es_low = fix_labels(sitk.GetArrayFromImage(sitk.ReadImage(path_low_seg_input_ES)))
            ####################################################################

            ####################################################################
            hight_ed_pred_target_seg = pre.getRestoreImg(cpu_hight_ed_pred_seg_target[0],gt_seg_ed_hight)
            hight_es_pred_target_seg = pre.getRestoreImg(cpu_hight_es_pred_seg_target[0],gt_seg_es_hight)
            ####################################################################

            ####################################################################
            hight_ed_pred_target_img = pre.getRestoreImg(cpu_hight_ed_pred_img_target[0],gt_seg_ed_hight)
            hight_es_pred_target_img = pre.getRestoreImg(cpu_hight_es_pred_img_target[0],gt_seg_es_hight)
            ####################################################################

            path_gt_UKBB_seg_ED = os.path.join(save_path,"gt_LR_seg_UKBB_ED.nii.gz")
            path_gt_UKBB_seg_ES = os.path.join(save_path,"gt_LR_seg_UKBB_ES.nii.gz")

            path_pred_UKBB_seg_ED = os.path.join(save_path,"pred_SR_seg_UKBB_ED.nii.gz")
            path_pred_UKBB_seg_ES = os.path.join(save_path,"pred_SR_seg_UKBB_ES.nii.gz")

            path_pred_UKBB_img_ED = os.path.join(save_path,"pred_SR_img_UKBB_ED.nii.gz")
            path_pred_UKBB_img_ES = os.path.join(save_path,"pred_SR_img_UKBB_ES.nii.gz")

            path_lr_pred_UKBB_seg_ED = os.path.join(save_path,"pred_resample_LR_seg_UKBB_ED.nii.gz")
            path_lr_pred_UKBB_seg_ES = os.path.join(save_path,"pred_resample_LR_seg_UKBB_ES.nii.gz")

            path_low_gray_input_ED_DA = os.path.join(save_path,"LR_DA_gray_ukbb_ED.nii.gz")
            path_low_gray_input_ES_DA = os.path.join(save_path,"LR_DA_gray_ukbb_ES.nii.gz")

            path_low_input_out_ED = os.path.join(save_path,"low_input_da_ED.nii.gz")
            path_low_input_out_ES = os.path.join(save_path,"low_input_da_ES.nii.gz")

            cpu_ed = pre.getRestoreImg(pad_target_img_ed_low,ed_gt_low)
            cpu_es = pre.getRestoreImg(pad_target_img_es_low,es_gt_low)
            
            pre.save_nifti_pred(cpu_ed,path_low_input_out_ED,path_low_gray_input_ED,sitk.sitkFloat64,False)
            pre.save_nifti_pred(cpu_es,path_low_input_out_ES,path_low_gray_input_ES,sitk.sitkFloat64,False)


            ####################################################################
            pre.save_nifti_pred(hight_ed_pred_target_seg,path_pred_UKBB_seg_ED,path_hight_seg_input_ED,sitk.sitkFloat64,True)
            pre.save_nifti_pred(hight_es_pred_target_seg,path_pred_UKBB_seg_ES,path_hight_seg_input_ES,sitk.sitkFloat64,True)
            ####################################################################

            ####################################################################
            LR_pred_seg_ED = Resize(path_low_seg_input_ED,path_pred_UKBB_seg_ED)
            LR_pred_seg_ES = Resize(path_low_seg_input_ES,path_pred_UKBB_seg_ES)
            ####################################################################

            ####################################################################
            pre.save_nifti_pred(hight_ed_pred_target_img,path_pred_UKBB_img_ED,path_hight_seg_input_ED,sitk.sitkFloat64,False)
            pre.save_nifti_pred(hight_es_pred_target_img,path_pred_UKBB_img_ES,path_hight_seg_input_ES,sitk.sitkFloat64,False)
            ####################################################################

            ####################################################################
            pre.save_nifti_pred(LR_pred_seg_ED,path_lr_pred_UKBB_seg_ED,path_low_seg_input_ED,sitk.sitkFloat64,True)
            pre.save_nifti_pred(LR_pred_seg_ES,path_lr_pred_UKBB_seg_ES,path_low_seg_input_ES,sitk.sitkFloat64,True)
            ####################################################################

            ####################################################################
            endo_ed,myo_ed,rv_ed =  DiceEval(LR_pred_seg_ED,gt_seg_ed_low)
            endo_es,myo_es,rv_es =  DiceEval(LR_pred_seg_ES,gt_seg_es_low)
            ####################################################################

            ######################################
            ed_global_endo_mean.append(endo_ed[0])
            ed_global_endo_std.append(endo_ed[1])
            ######################################
            ed_global_myo_mean.append(myo_ed[0])
            ed_global_myo_std.append(myo_ed[1])
            ######################################
            ed_global_rv_mean.append(rv_ed[0])
            ed_global_rv_std.append(rv_ed[1])
            ######################################
            es_global_endo_mean.append(endo_es[0])
            es_global_endo_std.append(endo_es[1])
            ######################################
            es_global_myo_mean.append(myo_es[0])
            es_global_myo_std.append(myo_es[1])   
            ######################################  
            es_global_rv_mean.append(rv_es[0])
            es_global_rv_std.append(rv_es[1])
            #####################################

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

    mean_dice_ed = 100*((global_mean_endo_ed + global_mean_myo_ed + global_mean_rv_ed)/3)
    mean_dice_es = 100*((global_mean_endo_es + global_mean_myo_es + global_mean_rv_es)/3)

    return mean_dice_ed,mean_dice_es


##########################################
#              MODELS                    #
##########################################


def TrainMUNIT(pathData,pathCode,pathModel,pathSave,n_worker,typedata,typemodel,flag_noise):
    #https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/munit/munit.py
    n_epochs_decay = 20
    num_epochs = 110
    #############
    latent_dim = 64
    n_downsample = 2
    style_dim = 8
    lr = 0.0002
    #############
    n_residual = 3
    lambda_gan = 1
    lambda_id = 10
    lambda_style = 1
    lambda_cont = 1
    lambda_cyc = 0
    in_channel = 100
    out_channels = 100
    every_valid = 10
    #############
    beta_1=0.5 
    beta_2=0.999 
    out_channels=1
    batch_size=1
    ############
    global_ssim_val_ed  = []
    global_ssim_val_es  = []
    ############
    global_dice_val_es  = []
    global_dice_val_ed  = []
    ############
    global_psnr_val_ed  = []

    print("\n ... Load Gemini-GAN")
    jointGenerator = load(pathModel)
    

    criterion_recon = torch.nn.L1Loss()

    # Initialize encoders, generators and discriminators

    Enc1 = model.EncoderMUNIT(in_channel,style_dim)
    Dec1 = model.DecoderMUNIT(in_channel,style_dim)
    Enc2 = model.EncoderMUNIT(in_channel,style_dim)
    Dec2 = model.DecoderMUNIT(in_channel,style_dim)
    D1 = model.MultiDiscriminatorMUNIT(in_channel)
    D2 = model.MultiDiscriminatorMUNIT(in_channel)

    Enc1 = Enc1.cuda()
    Dec1 = Dec1.cuda()
    Enc2 = Enc2.cuda()
    Dec2 = Dec2.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),lr=lr,betas=(beta_1, beta_2),)

    optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=lr, betas=(beta_1, beta_2))
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=lr, betas=(beta_1, beta_2))

    # ----------
    #  Training
    # ----------

    # Adversarial ground truths
    
    valid  = 1
    fake   = 0
    Tensor = torch.cuda.FloatTensor 
    train_set    = TrainDataset(pathData,pathCode,100,typedata)
    data_loader  = DataLoader(dataset=train_set, num_workers=n_worker, batch_size=batch_size)
    
    for epoch in range(num_epochs):
        ###########
        Enc1.train()
        Dec1.train()
        Enc2.train()
        Dec2.train()
        D1.train()
        D2.train()
        ###########
        for batch_idx, batch_train in tqdm(enumerate(data_loader,1),total=len(data_loader)):

            X1,\
            X2,_ = batch_train

            X1[X1 != X1] = 0
            X2[X2 != X2] = 0

            X1,\
            X2 = var(X1), var(X2)
            


            # Sampled style codes
            style_1 = Variable(torch.randn(X1.size(0), style_dim, 1, 1).type(Tensor))
            style_2 = Variable(torch.randn(X1.size(0), style_dim, 1, 1).type(Tensor))


            # -------------------------------
            #  Train Encoders and Generators
            # -------------------------------

            optimizer_G.zero_grad()

            # Get shared latent representation
            a1,b1      = Enc1(X1)
            c_code_1   = a1[0]
            cf_1       = a1[1]
            s_code_1   = b1

            a2,b2      = Enc2(X2)
            c_code_2   = a2[0]
            cf_2       = a2[1]
            s_code_2   = b2

            # Reconstruct images
            X11 = Dec1(c_code_1, X1, cf_1, s_code_1)
            X22 = Dec2(c_code_2, X2, cf_2, s_code_2)

            # Translate images
            X21 = Dec1(c_code_2, X2, cf_2, style_1)
            X12 = Dec2(c_code_1, X1, cf_1, style_2)

            # Cycle translation
            a21,b21     = Enc1(X21)
            c_code_21   = a21[0]
            cf_21       = a21[1]
            s_code_21   = b21

            a12,b12     = Enc1(X12)
            c_code_12   = a12[0]
            cf_12       = a12[1]
            s_code_12   = b12

            X121 = Dec1(c_code_12, X1, cf_12, s_code_1) if lambda_cyc > 0 else 0
            X212 = Dec2(c_code_21, X2, cf_21, s_code_2) if lambda_cyc > 0 else 0

            # Losses
            loss_GAN_1 = lambda_gan   * D1.compute_loss(X21, valid)
            loss_GAN_2 = lambda_gan   * D2.compute_loss(X12, valid)
            loss_ID_1  = lambda_id    * criterion_recon(X11, X1)
            loss_ID_2  = lambda_id    * criterion_recon(X22, X2)
            loss_s_1   = lambda_style * criterion_recon(s_code_21, style_1)
            loss_s_2   = lambda_style * criterion_recon(s_code_12, style_2)
            loss_c_1   = lambda_cont  * criterion_recon(c_code_12, c_code_1.detach())
            loss_c_2   = lambda_cont  * criterion_recon(c_code_21, c_code_2.detach())
            loss_cyc_1 = lambda_cyc   * criterion_recon(X121, X1) if lambda_cyc > 0 else 0
            loss_cyc_2 = lambda_cyc   * criterion_recon(X212, X2) if lambda_cyc > 0 else 0

            # Total loss
            loss_G = (
                loss_GAN_1
                + loss_GAN_2
                + loss_ID_1
                + loss_ID_2
                + loss_s_1
                + loss_s_2
                + loss_c_1
                + loss_c_2
                + loss_cyc_1
                + loss_cyc_2
            )
        
            

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator 1
            # -----------------------

            optimizer_D1.zero_grad()

            loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.detach(), fake)

            loss_D1.backward()
            optimizer_D1.step()

            # -----------------------
            #  Train Discriminator 2
            # -----------------------

            optimizer_D2.zero_grad()

            loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.detach(), fake)

            loss_D2.backward()
            optimizer_D2.step()

        
        # Valid
        #if epoch > 0 and epoch % every_valid==0:

        ###########
        pathSaveEpoch = os.path.join(pathSave,"Epoch_"+str(epoch)) 
        mkdir_dir(pathSaveEpoch)
        ###########
        
        if typedata == "SG":
            dice,psnr,ssim = validFunction(Enc1,Dec1,Dec1,Dec1,Dec1,Dec1,jointGenerator,pathCode,pathData,pathSaveEpoch,typedata,typemodel,flag_noise)
            global_ssim_val_ed.append(ssim[0])
            global_ssim_val_es.append(ssim[1])
            global_psnr_val_ed.append(psnr[0])
            global_psnr_val_es.append(psnr[1])
            global_dice_val_ed.append(dice[0])
            global_dice_val_es.append(dice[1])
            plot.dice_plot(removeNaN(global_dice_val_ed),pathSave,"ES")
            plot.dice_plot(removeNaN(global_dice_val_es),pathSave,"ED")
            plot.psnr_plot(removeNaN(global_psnr_val_ed),pathSave,"ED")
            plot.psnr_plot(removeNaN(global_psnr_val_es),pathSave,"ES")
            plot.ssim_plot(removeNaN(global_ssim_val_ed),pathSave,"ED")
            plot.ssim_plot(removeNaN(global_ssim_val_es),pathSave,"ES")
        elif typedata == "UKBB":
            dice_ed,dice_es = validFunctionUKBB(Enc1,Dec1,Dec1,Dec1,Dec1,Dec1,jointGenerator,pathCode,pathData,pathSaveEpoch,typedata,typemodel,flag_noise)
            global_dice_val_ed.append(dice_ed)
            global_dice_val_es.append(dice_es)
            plot.dice_plot(removeNaN(global_dice_val_ed),pathSave,"ES")
            plot.dice_plot(removeNaN(global_dice_val_es),pathSave,"ED")
        torch.save(Enc1.state_dict(),os.path.join(pathSaveEpoch,"Encoder.pt"))
        torch.save(Dec1.state_dict(),os.path.join(pathSaveEpoch,"Decoder.pt"))
        

def TrainCycleGAN(pathData,pathCode,pathModel,pathSave,n_worker,typedata,typemodel,flag_noise):

    ############
    dtype = torch.cuda.FloatTensor
    num_epoch = 110
    lr = 0.0002
    beta_1 = 0.5 
    beta_2 = 0.999 
    in_channels = 100
    out_channels = 100
    lambda_cyc = 10
    lambda_id = 5
    G_noise = False
    batch_size = 1
    every_valid = 10
    ############

    ############
    global_ssim_val_ed  = []
    global_ssim_val_es  = []
    ############
    global_dice_val_es  = []
    global_dice_val_ed  = []
    ############
    global_psnr_val_ed  = []
    global_psnr_val_es  = []
    ############
    plot  = plots()
    ############

    # Models
    Gab = model.SimpleGenerator(in_channels, out_channels).type(dtype)
    Gba = model.SimpleGenerator(in_channels, out_channels).type(dtype)
    Da  = model.SimpleDiscriminator().type(dtype)
    Db  = model.SimpleDiscriminator().type(dtype)
    # Optimizers
    g_optimizer = torch.optim.Adam(itertools.chain(Gab.parameters(),Gba.parameters()), lr=lr, betas=(beta_1, beta_2))
    da_optimizer = torch.optim.Adam(Da.parameters(), lr=lr, betas=(beta_1, beta_2))
    db_optimizer = torch.optim.Adam(Db.parameters(), lr=lr, betas=(beta_1, beta_2))
    
    # Losses
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()

    print("\n ... Load Gemini-GAN")
    jointGenerator = load(pathModel)
    
    fake_a_buffer = model.ReplayBuffer()
    fake_b_buffer = model.ReplayBuffer()

    train_set    = TrainDataset(pathData,pathCode,out_channels,typedata)
    data_loader  = DataLoader(dataset=train_set, num_workers=n_worker, batch_size=batch_size)

    for epoch in range(num_epoch):
        ###########
        Gab.train()
        Gba.train()
        Da.train()
        Db.train()
        ###########
        for batch_idx, batch_train in tqdm(enumerate(data_loader,1),total=len(data_loader)):
        
            ###############################
            a_real,\
            b_real,_ = batch_train

            a_real[a_real != a_real] = 0
            b_real[b_real != b_real] = 0
            

            a_real[a_real != a_real] = 0
            b_real[b_real != b_real] = 0

            a_real,\
            b_real = var(a_real), var(b_real)

            # ------------------
            #  Train Generators
            # ------------------

            g_optimizer.zero_grad()

            # Identity loss
            loss_id_A = L1(Gba(a_real,G_noise), a_real)
            loss_id_B = L1(Gab(b_real,G_noise), b_real)

            loss_identity = (loss_id_A + loss_id_B)/2

            b_fake = Gab(a_real,G_noise)
            out_fake = Db(b_fake)
            
            valid = tocuda(Variable(torch.ones(out_fake.size())))
            fake = tocuda(Variable(torch.zeros(out_fake.size())))

            loss_GAN_ab = MSE(out_fake, valid)

            a_fake = Gab(b_real,G_noise)
            loss_GAN_ba = MSE(Da(a_fake), valid)

            
            loss_GAN = (loss_GAN_ab + loss_GAN_ba)/2

            # Cycle loss

            recov_a = Gba(b_fake,G_noise)

            loss_cycle_a = L1(recov_a, a_real)

            recov_b = Gab(a_fake,G_noise)

            loss_cycle_b = L1(recov_b, b_real)

            loss_cycle = (loss_cycle_a + loss_cycle_b)/2
            
            # Total loss

            loss_G = loss_GAN + lambda_cyc*loss_cycle + lambda_id*loss_identity

            loss_G.backward()

            g_optimizer.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            da_optimizer.zero_grad()
            
            # Real loss

            loss_real_a = MSE(Da(a_real),valid)

            a_fake_ = fake_a_buffer.push_and_pop(a_fake)

            loss_fake_a = MSE(Da(a_fake_.detach()),fake)

            # Total loss
            
            loss_Da = (loss_real_a + loss_fake_a)/2
            
            loss_Da.backward()
            da_optimizer.step()


            # -----------------------
            #  Train Discriminator B
            # -----------------------

            db_optimizer.zero_grad()

            # Real loss
            loss_real_b = MSE(Db(b_real), valid)
            b_fake_ = fake_b_buffer.push_and_pop(b_fake)
            loss_fake_b = MSE(Db(b_fake_.detach()),fake)

            # Total loss
            loss_Db = (loss_real_b + loss_fake_b)/2


            loss_Db.backward()
            db_optimizer.step()

        if epoch > 0 and epoch % every_valid==0:
            ###########
            pathSaveEpoch = os.path.join(pathSave,"Epoch_"+str(epoch)) 
            mkdir_dir(pathSaveEpoch)
            ###########
            # Valid
            if typedata == "SG":
                dice,psnr,ssim =  validFunction(Gab,Gab,Gab,Gab,Gab,Gab,jointGenerator,pathCode,pathData,pathSaveEpoch,typedata,typemodel,flag_noise)
                global_ssim_val_ed.append(ssim[0])
                global_ssim_val_es.append(ssim[1])
                global_psnr_val_ed.append(psnr[0])
                global_psnr_val_es.append(psnr[1])
                global_dice_val_ed.append(dice[0])
                global_dice_val_es.append(dice[1])
                plot.dice_plot(removeNaN(global_dice_val_ed),pathSave,"ES")
                plot.dice_plot(removeNaN(global_dice_val_es),pathSave,"ED")
                plot.psnr_plot(removeNaN(global_psnr_val_ed),pathSave,"ED")
                plot.psnr_plot(removeNaN(global_psnr_val_es),pathSave,"ES")
                plot.ssim_plot(removeNaN(global_ssim_val_ed),pathSave,"ED")
                plot.ssim_plot(removeNaN(global_ssim_val_es),pathSave,"ES")
            elif typedata == "UKBB":
                dice_ed,dice_es = validFunctionUKBB(E_source,E_target,G_source,G_target,Noise_source,Noise_target,jointGenerator,pathCode,pathData,pathSaveEpoch,typedata,typemodel,flag_noise)
                global_dice_val_ed.append(dice_ed)
                global_dice_val_es.append(dice_es)
                plot.dice_plot(removeNaN(global_dice_val_ed),pathSave,"ES")
                plot.dice_plot(removeNaN(global_dice_val_es),pathSave,"ED")
            torch.save(Gab.state_dict(),os.path.join(pathSaveEpoch,"Generator.pt"))


def TrainBicycleGAN(pathData,pathCode,pathModel,pathSave,n_worker,typedata,typemodel,flag_noise):
    # D_cVAE is discriminator for cVAE-GAN(encoded vector z).
    # D_cLR is discriminator for cLR-GAN(random vector z).
    # Both of D_cVAE and D_cLR has two discriminators which 
    # have different output size((14x14) and (30x30)).
    # Totally, we have for discriminators now.
    ############
    dtype = torch.cuda.FloatTensor
    z_dim=8
    num_epoch=110
    lr=0.0002
    beta_1=0.5 
    beta_2=0.999 
    lambda_latent=0.01 
    lambda_pixel=10
    lambda_kl = 0.01
    lambda_latent=0.5
    in_channels = 100
    out_channels = 100
    every_valid = 10
    G_noise = False
    ############
    batch_size = 1
    ############
    global_ssim_val_ed  = []
    global_ssim_val_es  = []
    ############
    global_dice_val_es  = []
    global_dice_val_ed  = []
    ############
    global_psnr_val_ed  = []
    global_psnr_val_es  = []
    ############
    plot  = plots()
    ##########
    # Models #
    ##########
    D_VAE = model.Discriminator().type(dtype)
    D_LR  = model.Discriminator().type(dtype)
    generator  = model.Generator(in_channels, out_channels,z_dim=z_dim).type(dtype)
    encoder    = model.Encoder(z_dim=z_dim).type(dtype)
    # Loss functions
    mae_loss = torch.nn.L1Loss()
    generator = generator.cuda()
    encoder.cuda()
    D_VAE = D_VAE.cuda()
    D_LR = D_LR.cuda()
    mae_loss.cuda()

    ############
    # Optimizers
    ############
    optimizer_D_VAE = optim.Adam(D_VAE.parameters(), lr=lr, betas=(beta_1, beta_2))
    optimizer_D_LR = optim.Adam(D_LR.parameters(), lr=lr, betas=(beta_1, beta_2))
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
    optimizer_E = optim.Adam(encoder.parameters(), lr=lr, betas=(beta_1, beta_2))
    Tensor = torch.cuda.FloatTensor 
    ############    
    train_set    = TrainDataset(pathData,pathCode,out_channels,typedata)
    data_loader  = DataLoader(dataset=train_set, num_workers=n_worker, batch_size=batch_size)
    ############
    print("\n ... Load Gemini-GAN")
    jointGenerator = load(pathModel)
    ############

    for epoch in range(num_epoch):
    
        ###########
        D_VAE.train()
        D_LR.train()
        generator.train()
        encoder.train()
        ###########
    
        for batch_idx, batch_train in tqdm(enumerate(data_loader,1),total=len(data_loader)):
            
            ######################################
            # Retrive numpy data from dataLoder  #
            ######################################
            
            ###############################
            img,\
            ground_truth,_ = batch_train
            ###############################
            img[img != img] = 0
            ground_truth [ground_truth != ground_truth] = 0
            ###############################

            real_A, real_B = var(img), var(ground_truth)

            valid = Variable(torch.Tensor(np.ones((real_A.size(0),1))), requires_grad=False).cuda()
            fake  = Variable(torch.Tensor(np.zeros((real_B.size(0), 1))), requires_grad=False).cuda()

            # -------------------------------
            #  Train Generator and Encoder
            # -------------------------------

            optimizer_E.zero_grad()
            optimizer_G.zero_grad()

            # ----------
            # cVAE-GAN
            # ----------

            # Produce output using encoding of B (cVAE-GAN)
            mu, logvar = encoder(real_B)
            encoded_z = model.reparameterizationBicycleGAN(mu, logvar)
            fake_B = generator(real_A, encoded_z, False)

            # Pixelwise loss of translated image by VAE
            loss_pixel = mae_loss(fake_B, real_B)
            # Kullback-Leibler divergence of encoded B
            loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
            # Adversarial loss
            loss_VAE_GAN = mae_loss(fake_B, valid)

            # ---------
            # cLR-GAN
            # ---------

            # Produce output using sampled z (cLR-GAN)
            sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), z_dim))))
            _fake_B = generator(real_A, sampled_z, False)
            # cLR Loss: Adversarial loss
            loss_LR_GAN = mae_loss(_fake_B, valid)

            # ----------------------------------
            # Total Loss (Generator + Encoder)
            # ----------------------------------

            loss_GE = loss_VAE_GAN + loss_LR_GAN + lambda_pixel * loss_pixel + lambda_kl * loss_kl

            loss_GE.backward(retain_graph=True)
            optimizer_E.step()

            # ---------------------
            # Generator Only Loss
            # ---------------------

            # Latent L1 loss
            _mu, _ = encoder(_fake_B)
            loss_latent = lambda_latent * mae_loss(_mu, sampled_z)

            loss_latent.backward()
            optimizer_G.step()

            # ----------------------------------
            #  Train Discriminator (cVAE-GAN)
            # ----------------------------------

            optimizer_D_VAE.zero_grad()

            loss_D_VAE = mae_loss(real_B, valid) + mae_loss(fake_B.detach(), fake)

            loss_D_VAE.backward()
            optimizer_D_VAE.step()

            # ---------------------------------
            #  Train Discriminator (cLR-GAN)
            # ---------------------------------

            optimizer_D_LR.zero_grad()

            loss_D_LR = mae_loss(real_B, valid) + mae_loss(_fake_B.detach(), fake)

            loss_D_LR.backward()
            optimizer_D_LR.step()


        if epoch > 0 and epoch % every_valid==0:
            ###########
            pathSaveEpoch = os.path.join(pathSave,"Epoch_"+str(epoch)) 
            mkdir_dir(pathSaveEpoch)
            ###########
            # Valid
            dice,psnr,ssim = validFunction(generator,generator,generator,generator,generator,generator,jointGenerator,pathCode,pathData,pathSaveEpoch,typedata,typemodel,flag_noise)
            global_ssim_val_ed.append(ssim[0])
            global_ssim_val_es.append(ssim[1])
            global_psnr_val_ed.append(psnr[0])
            global_psnr_val_es.append(psnr[1])
            global_dice_val_ed.append(dice[0])
            global_dice_val_es.append(dice[1])
            plot.dice_plot(removeNaN(global_dice_val_ed),pathSave,"ES")
            plot.dice_plot(removeNaN(global_dice_val_es),pathSave,"ED")
            plot.psnr_plot(removeNaN(global_psnr_val_ed),pathSave,"ED")
            plot.psnr_plot(removeNaN(global_psnr_val_es),pathSave,"ES")
            plot.ssim_plot(removeNaN(global_ssim_val_ed),pathSave,"ED")
            plot.ssim_plot(removeNaN(global_ssim_val_es),pathSave,"ES")
            torch.save(generator.state_dict(),os.path.join(pathSaveEpoch,"Generator.pt"))


def retrain(retrain_model,lr_input,sr_seg,sr_img,loss_seg,loss_img):
    retrain_model.train()
    retrain_model.zero_grad()
    r_output_hr,\
    r_output_seg_hr   = retrain_model(lr_input)
    g_loss            = loss_img(r_output_hr[0],sr_img[0]) + loss_seg(r_output_seg_hr,sr_seg.unsqueeze(0)) 
    g_loss.backward()
    optimizer_G.step()
    return retrain_model

def V_AMA(pathData,pathCode,pathModel,pathSave,n_worker,typedata,typemodel,flag_noise):

    num_epoch = 1000
    every_valid = 1
    lr = 0.0002
    beta_1=0.5 
    beta_2=0.999 
    loss_seg  = nn.CrossEntropyLoss().to ("cuda")
    loss_img  = nn.MSELoss().to ("cuda")

    print("\n ... Load Gemini-GAN")
    jointGenerator = load(pathModel)


    ############
    global_ssim_val_ed  = []
    global_ssim_val_es  = []
    ############
    global_dice_val_es  = []
    global_dice_val_ed  = []
    ############
    global_psnr_val_ed  = []
    global_psnr_val_es  = []
    ############

    plot  = plots()    
    E_source   = model.EncoderVAE(100)
    E_source   = E_source.float().cuda()

    E_target   = model.EncoderVAE(100)
    E_target   = E_target.float().cuda()

    G_source = model.DecoderBA(100)
    G_source = G_source.float().cuda()

    G_target = model.DecoderAB(100)
    G_target = G_target.float().cuda()

    Noise_source = model.GetMeanStd(1)
    Noise_source = Noise_source.cuda()

    Noise_target = model.GetMeanStd(1)
    Noise_target = Noise_target.cuda()

    d_source  = model.DiscriminatorDomain1(100,1)
    d_source  = d_source.float().cuda()

    d_target  = model.DiscriminatorDomain1(100,1)
    d_target  = d_target.float().cuda()

    optimizer_G_source = torch.optim.Adam(itertools.chain(E_source.parameters(),G_source.parameters(),Noise_source.parameters()),betas=(beta_1, beta_2),lr=lr)
    
    optimizer_G_target = torch.optim.Adam(itertools.chain(E_target.parameters(),G_target.parameters(),Noise_target.parameters()),betas=(beta_1, beta_2),lr=lr)

    optimizer_D_target = torch.optim.Adam(d_target.parameters(),betas=(beta_1, beta_2),lr=lr)

    optimizer_D_source = torch.optim.Adam(d_source.parameters(),betas=(beta_1, beta_2),lr=lr)

    MSE = nn.MSELoss().float().cuda()
    L1 = nn.L1Loss().float().cuda()
    
    valid_set    = TrainDataset(pathData,pathCode,100,typedata)

    data_loader  = DataLoader(dataset=valid_set, num_workers=n_worker, batch_size=1)

    MSE = nn.MSELoss().cuda()

    for epoch in range(num_epoch):

        ###########
        E_source.train()
        E_target.train()
        G_source.train()
        G_target.train()
        Noise_source.train()
        Noise_target.train()
        d_target.train()
        d_source.train()
        ###########
        

        print("\n ... Epoch number " + str(epoch))
        for batch_idx, batch_train in tqdm(enumerate(data_loader,1),total=len(data_loader)):
            
            
            optimizer_G_target.zero_grad()
            valid_t =  Variable(torch.from_numpy(np.ones((1,1))).float().cuda())
            fake_t  =  Variable(torch.from_numpy(np.zeros((1,1))).float().cuda())

            ###############################
            a_real,\
            b_real,\
            gt_lr_seg = batch_train
            ###############################

            a_real[a_real != a_real] = 0
            b_real [b_real != b_real] = 0
            gt_lr_seg [gt_lr_seg != gt_lr_seg] = 0
            a_real,\
            b_real = var(a_real), var(b_real)
            
            # 1. Target Optimisation
            in_data_a = a_real + b_real
            et,ft     = E_target(in_data_a)
            mu_t, log_var_t = Noise_source(et)
            std_t      = torch.exp(log_var_t/2)
            random_z_t = var(torch.randn(1,320000))
            random_z_t = (random_z_t*std_t) + mu_t
            random_z_t = random_z_t.view(et.size())
            recon_t  = G_target(random_z_t,b_real,ft)
            gan_loss_t =  MSE(d_target(recon_t),valid_t) 
            KL_div_t = torch.sum(0.5 * (mu_t ** 2 + torch.exp(log_var_t)  - log_var_t - 1))
            g_idt_t =  MSE(b_real,recon_t) + L1(b_real,recon_t) + 0.5*KL_div_t + 0.05*gan_loss_t
            g_idt_t.backward()
            optimizer_G_target.step()
            optimizer_D_target.zero_grad()
            fake_dt  = d_target(recon_t.detach())
            valid_dt = d_target(b_real)
            dt_fake_loss = MSE(fake_dt,fake_t)
            dt_real_loss = MSE(valid_dt,valid_t) 
            dt_loss_dom = (dt_fake_loss + dt_real_loss)/2
            dt_loss_dom.backward()
            optimizer_D_target.step()

            # 2. Source Optimisation
            optimizer_G_source.zero_grad()
            valid_s =  Variable(torch.from_numpy(np.ones((1,1))).float().cuda())
            fake_s  =  Variable(torch.from_numpy(np.zeros((1,1))).float().cuda())
            es,fs  = E_source(a_real)
            mu_s, log_var_s = Noise_target(es)
            std_s      = torch.exp(log_var_s/2)
            random_z_s = var(torch.randn(1,320000))
            random_z_s = random_z_s*(std_s+std_t.detach()) + (mu_s+mu_t.detach())
            random_z_s = random_z_s.view(es.size())
            fm = [fs[0]+ft[0].detach(),fs[1]+ft[1].detach(),fs[2]+ft[2].detach(),fs[3]+ft[3].detach()]
            recon_s  = G_source(random_z_s,a_real,fm) 
            gan_loss_s =  MSE(d_source(recon_s),valid_s) 
            KL_div_s = torch.sum(0.5 * ((mu_s + mu_t.detach())** 2 + torch.exp(log_var_s) + torch.exp(log_var_t.detach())  - log_var_t.detach() - log_var_s - 1))
            g_idt_s =  MSE(a_real,recon_s) + L1(a_real,recon_s) + 0.5*KL_div_s + 0.05*gan_loss_s
            g_idt_s.backward()
            optimizer_G_source.step()
            optimizer_D_source.zero_grad()
            fake_ds  = d_source(recon_s.detach())
            valid_ds = d_source(b_real)
            ds_fake_loss = MSE(fake_ds,fake_s)
            ds_real_loss = MSE(valid_ds,valid_s) 
            ds_loss_dom  = (ds_fake_loss + ds_real_loss)/2
            ds_loss_dom.backward()
            optimizer_D_source.step()
                
        if epoch > 0 and epoch % every_valid==0:
            pathSaveEpoch = os.path.join(pathSave,"Epoch_"+str(epoch)) 
            mkdir_dir(pathSaveEpoch)
            # Valid
            if typedata == "SG":
                dice,psnr,ssim  = validFunction(E_source,E_target,G_source,G_target,Noise_source,Noise_target,jointGenerator,pathCode,pathData,pathSaveEpoch,typedata,typemodel,flag_noise)
                global_ssim_val_ed.append(ssim[0])
                global_ssim_val_es.append(ssim[1])
                global_psnr_val_ed.append(psnr[0])
                global_psnr_val_es.append(psnr[1])
                global_dice_val_ed.append(dice[0])
                global_dice_val_es.append(dice[1])
                plot.dice_plot(removeNaN(global_dice_val_ed),pathSave,"ES")
                plot.dice_plot(removeNaN(global_dice_val_es),pathSave,"ED")
                plot.psnr_plot(removeNaN(global_psnr_val_ed),pathSave,"ED")
                plot.psnr_plot(removeNaN(global_psnr_val_es),pathSave,"ES")
                plot.ssim_plot(removeNaN(global_ssim_val_ed),pathSave,"ED")
                plot.ssim_plot(removeNaN(global_ssim_val_es),pathSave,"ES")
            elif typedata == "UKBB": 
                dice_ed,dice_es = validFunctionUKBB(E_source,E_target,G_source,G_target,Noise_source,Noise_target,jointGenerator,pathCode,pathData,pathSaveEpoch,typedata,typemodel,flag_noise)
                global_dice_val_ed.append(dice_ed)
                global_dice_val_es.append(dice_es)
                plot.dice_plot(removeNaN(global_dice_val_ed),pathSave,"ES")
                plot.dice_plot(removeNaN(global_dice_val_es),pathSave,"ED")
            torch.save(E_source.state_dict(),os.path.join(pathSaveEpoch,"E_source.pt"))
            torch.save(E_target.state_dict(),os.path.join(pathSaveEpoch,"E_target.pt"))
            torch.save(G_source.state_dict(),os.path.join(pathSaveEpoch,"G_source.pt"))
            torch.save(Noise_source.state_dict(),os.path.join(pathSaveEpoch,"Noise_source.pt"))
            torch.save(Noise_target.state_dict(),os.path.join(pathSaveEpoch,"Noise_target.pt"))
            torch.save(jointGenerator.state_dict(),os.path.join(pathSaveEpoch,"Generator.pt"))
            print("\n ... Model saved done.")

def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

if __name__ == '__main__':
    typedata   = "SG" # SG or UKBB
    typemodel  = "V_AMA" #BicycleGAN, CycleGAN, UNIT, MUNIT, V_AMA
    n_worker   = 4
    
    if typedata == "SG":
        pathData = "" # Folder SG dataset
    elif typedata =="UKBB":
        pathData = "" # Folder UKBB dataset
    
    pathCode  = "" # Folder where is the list of patients (named dataset_txt)         
    pathsave  = "" # Folder where to save
    pathModel = "" # Folder at Gemini-GAN model 
    n_worker  = 4  # Number of CPU worker

    today     =  date.today()
    pathSave  =  os.path.join(pathsave,"DA"+ typedata + "_" + str(today))
    mkdir_dir(pathSave)
    pathSave  =  os.path.join(pathSave,typemodel) 
    mkdir_dir(pathSave)


    if   typemodel == "BicycleGAN":
            TrainBicycleGAN(pathData,pathCode,pathModel,pathSave,n_worker,typedata,typemodel,False) 

    elif typemodel == "CycleGAN":
            TrainCycleGAN(pathData,pathCode,pathModel,pathSave,n_worker,typedata,typemodel,False) 
    
    elif typemodel == "MUNIT":
            TrainMUNIT(pathData,pathCode,pathModel,pathSave,n_worker,typedata,typemodel,False) 
    
    elif typemodel == "V_AMA":
            V_AMA(pathData,pathCode,pathModel,pathSave,n_worker,typedata,typemodel,False)



