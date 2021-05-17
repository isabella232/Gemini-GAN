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
import copy
from PIL import Image, ImageEnhance 
import warnings
from torchvision import transforms, utils
warnings.simplefilter("ignore",  category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 


###############################################################################
# Before starting the code please configure the paths at the end of the code  #
###############################################################################


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
            # In case of DM we need to remove oulier as due sensibility of SSIM
            #if out_ssim >=0.999:
            #    continue
            if  not math.isnan(out_ssim) or out_ssim == -np.inf or out_ssim == np.inf or out_ssim <0:
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


def DoubleInference2d(input_img):
    genscan_data = []
    ukbb_data = []
    pre = preprocessing(400,1)
    for i in range(input_img.shape[0]): 
        with torch.no_grad():
            pad_input = pre.pad_img(input_img[i])
            input_img_cuda = Variable(get_torch(pad_input).to("cuda").unsqueeze(0).float())
            output_genscan,output_ukbb = generator(input_img_cuda,False)
            genscan_data.append(output_genscan[0][0])
            ukbb_data.append(output_ukbb[0][0])
    genscan_output = torch.stack(genscan_data)
    ukbb_output = torch.stack(ukbb_data)
    return genscan_output.cpu().data.numpy(),\
           ukbb_output.cpu().data.numpy()

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
        elif typedata == "CycleGAN":
            output_img_cuda = Es(input_img_cuda,False)    
        
        elif typedata == "BicycleGAN": 
            random_z = var(torch.randn(1,8))
            output_img_cuda = Es(input_img_cuda, random_z,False)  

        elif typedata == "V-AMA": 
            et,ft            = Et(input_img_cuda)
            es,fs            = Es(input_img_cuda)
            mu_t, log_var_t  = Nt(et)
            mu_s, log_var_s  = Ns(es)
            std_t            = torch.exp(log_var_t / 2)
            std_s            = torch.exp(log_var_s / 2)
            random_z_t       = var(torch.randn(1, 320000))
            random_z_t       = random_z_t*(std_s+std_t) + (mu_t+mu_s)
            random_z_t       = random_z_t.view(es.size())
            fm               = [fs[0]+ft[0],fs[1]+ft[1],fs[2]+ft[2],fs[3]+ft[3]]
            output_img_cuda  = Gs(random_z_t,input_img_cuda,fm)

    return output_img_cuda.cpu().numpy()[0]

def testFunction_no_dm(jointGenerator,pathCode,pathData,pathSave,typemodel,flag_noise):

    #######################
    ed_global_endo_mean = []
    ed_global_myo_mean = []
    ed_global_rv_mean = []
    ed_global_endo_std = []
    ed_global_myo_std = []
    ed_global_rv_std = []
    es_global_endo_mean = []
    es_global_myo_mean = []
    es_global_rv_mean = []
    es_global_endo_std = []
    es_global_myo_std = []
    es_global_rv_std = []
    #######################
    ed_mean_psnr = []
    ed_std_psnr = []
    es_mean_psnr = []
    es_std_psnr = []
    ed_mean_ssim = []
    ed_std_ssim = []
    es_mean_ssim = []
    es_std_ssim = []
    #######################

    patients = open_txt(os.path.join(pathCode,"dataset_txt","DA","test.txt"))
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
            
            path_low_gray_input_ED = os.path.join(pathData,patient,"low_gray_SG_ED.nii.gz")
            path_low_gray_input_ES = os.path.join(pathData,patient,"low_gray_SG_ES.nii.gz")

            path_hight_gray_input_ED = os.path.join(pathData,patient,"hight_gray_SG_ED.nii.gz")
            path_hight_gray_input_ES = os.path.join(pathData,patient,"hight_gray_SG_ES.nii.gz")

            path_hight_seg_input_ED = os.path.join(pathData,patient,"hight_seg_SG_ED.nii.gz")
            path_hight_seg_input_ES = os.path.join(pathData,patient,"hight_seg_SG_ES.nii.gz")
            
            ed_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ED))
            es_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ES))

            pad_target_img_ed_low = pre.pad_seq(ed_gt_low)
            pad_target_img_es_low = pre.pad_seq(es_gt_low)


            ####################################################################
            gt_seg_ed_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ED))
            gt_seg_es_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ES))
            ####################################################################


            if flag_noise:
                path_low_input_noise_ED = os.path.join(save_path,"low_input_noise_ED.nii.gz")
                path_low_input_noise_ES = os.path.join(save_path,"low_input_noise_ES.nii.gz")
                pad_target_img_ed_low   = brightness(pad_target_img_ed_low)
                pad_target_img_es_low   = brightness(pad_target_img_es_low)
                cuda_img_ed_target_cuda = Variable(get_torch(pad_target_img_ed_low).to("cuda").float())
                cuda_img_es_target_cuda = Variable(get_torch(pad_target_img_es_low).to("cuda").float())
                scanner_noise = torch.FloatTensor(cuda_img_ed_target_cuda.size()).normal_(0,1).cuda()
                cuda_img_ed_target_cuda = torch.add(cuda_img_ed_target_cuda,0.05*scanner_noise)
                cuda_img_es_target_cuda = torch.add(cuda_img_es_target_cuda,0.05*scanner_noise)
                # save test 
                cpu_ed = cuda_img_ed_target_cuda.cpu().data.numpy()[0]
                cpu_es = cuda_img_es_target_cuda.cpu().data.numpy()[0]
                cpu_ed_t = pre.getRestoreImg(cpu_ed,ed_gt_low)
                cpu_es_t = pre.getRestoreImg(cpu_es,es_gt_low)
                pre.save_nifti_pred(cpu_ed_t,path_low_input_noise_ED,path_low_gray_input_ED,sitk.sitkFloat64,False)
                pre.save_nifti_pred(cpu_es_t,path_low_input_noise_ES,path_low_gray_input_ES,sitk.sitkFloat64,False)
            else:
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
    es_std_psnr_global = np.std(removeNaN(es_std_psnr))
    ###
    ed_mean_ssim_global = np.mean(removeNaN(ed_mean_ssim))
    ed_std_ssim_global = np.std(removeNaN(ed_std_ssim))
    ###
    es_mean_ssim_global = np.mean(removeNaN(es_mean_ssim))
    es_std_ssim_global = np.std(removeNaN(es_std_ssim))
    ###

    str_save = "... Total patients " + str(len(ed_mean_psnr)) + " Model Name " + typemodel      + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ED  "      + str(global_mean_endo_ed) + "/" + str(global_std_endo_ed) + "\n" + \
               " ... MYO ED  "       + str(global_mean_myo_ed)  + "/" + str(global_std_myo_ed)  + "\n" + \
               " ... RV ED  "        + str(global_mean_rv_ed)   + "/" + str(global_std_rv_ed)   + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ES  "      + str(global_mean_endo_es) + "/" + str(global_std_endo_es) + "\n" + \
               " ... MYO ES  "       + str(global_mean_myo_es)  + "/" + str(global_std_myo_es)  + "\n" + \
               " ... RV ES  "        + str(global_mean_rv_es)   + "/" + str(global_std_rv_es)   + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... PSNR ED  "      + str(ed_mean_psnr_global) + "/" + str(ed_std_psnr_global) + "\n" + \
               " ... SSIM ED  "      + str(ed_mean_ssim_global) + "/" + str(ed_std_ssim_global) + "\n" + \
               " ... PSNR ES  "      + str(es_mean_psnr_global) + "/" + str(es_std_psnr_global) + "\n" + \
               " ... SSIM ES  "      + str(es_mean_ssim_global) + "/" + str(es_std_ssim_global)
    text_file = open(os.path.join(pathSave,typemodel + "_table.txt"), "w")
    text_file.write(str_save)
    text_file.close()




def testFunctionUKBB(Es,Et,Gs,Gt,Ns,Nt,generator_retrain,pathCode,pathData,pathSave,typedata,typemodel,flag_noise):   
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
    patients  = open_txt(os.path.join(pathCode,"dataset_txt","test.txt"))
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

    str_save = "... Total patients " + str(len(ed_mean_psnr)) + " Model Name " + typemodel      + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ED  "      + str(global_mean_endo_ed) + "/" + str(global_std_endo_ed) + "\n" + \
               " ... MYO ED  "       + str(global_mean_myo_ed)  + "/" + str(global_std_myo_ed)  + "\n" + \
               " ... RV ED  "        + str(global_mean_rv_ed)   + "/" + str(global_std_rv_ed)   + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ES  "      + str(global_mean_endo_es) + "/" + str(global_std_endo_es) + "\n" + \
               " ... MYO ES  "       + str(global_mean_myo_es)  + "/" + str(global_std_myo_es)  + "\n" + \
               " ... RV ES  "        + str(global_mean_rv_es)   + "/" + str(global_std_rv_es)   + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... PSNR ED  "      + str(ed_mean_psnr_global) + "/" + str(ed_std_psnr_global) + "\n" + \
               " ... SSIM ED  "      + str(ed_mean_ssim_global) + "/" + str(ed_std_ssim_global) + "\n" + \
               " ... PSNR ES  "      + str(es_mean_psnr_global) + "/" + str(es_std_psnr_global) + "\n" + \
               " ... SSIM ES  "      + str(es_mean_ssim_global) + "/" + str(es_std_ssim_global)
    text_file = open(os.path.join(pathSave,typemodel + "_table.txt"), "w")
    text_file.write(str_save)
    text_file.close()



def testFunction_NODM(jointGenerator,pathCode,pathData,pathSave,typemodel,flag_noise):
    
    #######################
    ed_global_endo_mean = []
    ed_global_myo_mean = []
    ed_global_rv_mean = []
    ed_global_endo_std = []
    ed_global_myo_std = []
    ed_global_rv_std = []
    es_global_endo_mean = []
    es_global_myo_mean = []
    es_global_rv_mean = []
    es_global_endo_std = []
    es_global_myo_std = []
    es_global_rv_std = []
    #######################
    ed_mean_psnr = []
    ed_std_psnr = []
    es_mean_psnr = []
    es_std_psnr = []
    ed_mean_ssim = []
    ed_std_ssim = []
    es_mean_ssim = []
    es_std_ssim = []
    #######################

    patients = open_txt(os.path.join(pathCode,"dataset_txt","DA","test.txt"))
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
            
            path_low_gray_input_ED = os.path.join(pathData,patient,"low_gray_SG_ED.nii.gz")
            path_low_gray_input_ES = os.path.join(pathData,patient,"low_gray_SG_ES.nii.gz")

            path_hight_gray_input_ED = os.path.join(pathData,patient,"hight_gray_SG_ED.nii.gz")
            path_hight_gray_input_ES = os.path.join(pathData,patient,"hight_gray_SG_ES.nii.gz")

            path_hight_seg_input_ED = os.path.join(pathData,patient,"hight_seg_SG_ED.nii.gz")
            path_hight_seg_input_ES = os.path.join(pathData,patient,"hight_seg_SG_ES.nii.gz")
            
            ed_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ED))
            es_gt_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ES))

            pad_target_img_ed_low = pre.pad_seq(ed_gt_low)
            pad_target_img_es_low = pre.pad_seq(es_gt_low)

            ####################################################################
            gt_seg_ed_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ED))
            gt_seg_es_hight = sitk.GetArrayFromImage(sitk.ReadImage(path_hight_seg_input_ES))
            ####################################################################

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
            hight_ed_pred_target_seg = pre.getRestoreImg(cpu_hight_ed_pred_seg_target[0],gt_seg_ed_hight)
            hight_es_pred_target_seg = pre.getRestoreImg(cpu_hight_es_pred_seg_target[0],gt_seg_es_hight)
            ####################################################################

            ####################################################################
            hight_ed_pred_target_img = pre.getRestoreImg(cpu_hight_ed_pred_img_target[0],gt_seg_ed_hight)
            hight_es_pred_target_img = pre.getRestoreImg(cpu_hight_es_pred_img_target[0],gt_seg_es_hight)
            ####################################################################
            
            
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
            pre.save_nifti_pred(hight_ed_pred_target_img,path_pred_SG_img_ED,path_hight_seg_input_ED,sitk.sitkFloat64,False)
            pre.save_nifti_pred(hight_es_pred_target_img,path_pred_SG_img_ES,path_hight_seg_input_ES,sitk.sitkFloat64,False)
            ####################################################################
            

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
    out = resample.Execute(sitk.Cast(image_SR, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(out)



def testFunction_no_dm_ukbb(jointGenerator,pathCode,pathData,pathSave,typemodel,flag_noise):

    jointGenerator.eval()

    #######################
    ed_global_endo_mean = []
    ed_global_myo_mean = []
    ed_global_rv_mean = []
    ed_global_endo_std = []
    ed_global_myo_std = []
    ed_global_rv_std = []
    es_global_endo_mean = []
    es_global_myo_mean = []
    es_global_rv_mean = []
    es_global_endo_std = []
    es_global_myo_std = []
    es_global_rv_std = []
    #######################
    ed_mean_psnr = []
    ed_std_psnr = []
    es_mean_psnr = []
    es_std_psnr = []
    ed_mean_ssim = []
    ed_std_ssim = []
    es_mean_ssim = []
    es_std_ssim = []
    #######################

    patients = open_txt(os.path.join(pathCode,"dataset_txt","test.txt"))
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
            
            pad_target_img_ed_low = pre.pad_seq(input_img_ed_low)
            pad_target_img_es_low = pre.pad_seq(input_img_es_low)

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
            gt_seg_ed_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_seg_input_ED))
            gt_seg_es_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_seg_input_ES))
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

            path_lr_pred_UKBB_seg_ED = os.path.join(save_path,"pred_LR_seg_UKBB_ED.nii.gz")
            path_lr_pred_UKBB_seg_ES = os.path.join(save_path,"pred_LR_seg_UKBB_ES.nii.gz")

            ####################################################################
            pre.save_nifti_pred(gt_seg_ed_low,path_gt_UKBB_seg_ED,path_low_seg_input_ED,sitk.sitkFloat64,True)
            pre.save_nifti_pred(gt_seg_es_low,path_gt_UKBB_seg_ES,path_low_seg_input_ES,sitk.sitkFloat64,True)
            ####################################################################

            ####################################################################
            pre.save_nifti_pred(hight_ed_pred_target_seg,path_pred_UKBB_seg_ED,path_hight_seg_input_ED,sitk.sitkFloat64,True)
            pre.save_nifti_pred(hight_es_pred_target_seg,path_pred_UKBB_seg_ES,path_hight_seg_input_ES,sitk.sitkFloat64,True)
            ####################################################################

            ####################################################################
            pre.save_nifti_pred(hight_ed_pred_target_img,path_pred_UKBB_img_ED,path_hight_seg_input_ED,sitk.sitkFloat64,False)
            pre.save_nifti_pred(hight_es_pred_target_img,path_pred_UKBB_img_ES,path_hight_seg_input_ES,sitk.sitkFloat64,False)
            ####################################################################
            
            ####################################################################
            LR_pred_seg_ED = Resize(path_low_seg_input_ED,path_pred_UKBB_seg_ED)
            LR_pred_seg_ES = Resize(path_low_seg_input_ES,path_pred_UKBB_seg_ES)
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


    str_save = "... Total patients " + str(len(ed_mean_psnr)) + " Model Name " + typemodel      + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ED  "      + str(global_mean_endo_ed) + "/" + str(global_std_endo_ed) + "\n" + \
               " ... MYO ED  "       + str(global_mean_myo_ed)  + "/" + str(global_std_myo_ed)  + "\n" + \
               " ... RV ED  "        + str(global_mean_rv_ed)   + "/" + str(global_std_rv_ed)   + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ES  "      + str(global_mean_endo_es) + "/" + str(global_std_endo_es) + "\n" + \
               " ... MYO ES  "       + str(global_mean_myo_es)  + "/" + str(global_std_myo_es)  + "\n" + \
               " ... RV ES  "        + str(global_mean_rv_es)   + "/" + str(global_std_rv_es)   + "\n" + \
               " -------------------------------------------------------------------------  "  

    text_file = open(os.path.join(pathSave,typemodel + "_table.txt"), "w")
    text_file.write(str_save)
    text_file.close()



def testFunction(Es,Et,Gs,Gt,Ns,Nt,jointGenerator,pathCode,pathData,pathSave,typemodel,flag_noise):   
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

    patients  = open_txt(os.path.join(pathCode,"dataset_txt","DA","test.txt"))
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
            
            path_low_gray_input_ED = os.path.join(pathData,patient,"low_gray_SG_ED.nii.gz")
            path_low_gray_input_ES = os.path.join(pathData,patient,"low_gray_SG_ES.nii.gz")

            path_hight_gray_input_ED = os.path.join(pathData,patient,"hight_gray_SG_ED.nii.gz")
            path_hight_gray_input_ES = os.path.join(pathData,patient,"hight_gray_SG_ES.nii.gz")

            path_hight_seg_input_ED = os.path.join(pathData,patient,"hight_seg_SG_ED.nii.gz")
            path_hight_seg_input_ES = os.path.join(pathData,patient,"hight_seg_SG_ES.nii.gz")

            input_img_ed_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ED))
            input_img_es_low = sitk.GetArrayFromImage(sitk.ReadImage(path_low_gray_input_ES))
            
            if flag_noise:                 
                list_input_img_ed_low = []
                for i in range(input_img_ed_low.shape[0]):
                    a_real_br_ed  = torch.from_numpy(brightness(input_img_ed_low[i]))
                    scanner_noise_ed = torch.FloatTensor(a_real_br_ed.size()).normal_(0,1)
                    img_ed_target_noise = torch.add(a_real_br_ed,0.05*scanner_noise_ed)
                    list_input_img_ed_low.append(img_ed_target_noise)
                
                list_input_img_es_low = []
                for i in range(input_img_ed_low.shape[0]):
                    a_real_br_es  = torch.from_numpy(brightness(input_img_es_low[i]))
                    scanner_noise_es = torch.FloatTensor(a_real_br_es.size()).normal_(0,1)
                    img_es_target_noise = torch.add(a_real_br_es,0.05*scanner_noise_es)
                    list_input_img_es_low.append(img_es_target_noise)

            
                nosie_input_img_ed_low =  torch.stack(list_input_img_ed_low).cpu().data.numpy()
                nosie_input_img_es_low =  torch.stack(list_input_img_es_low).cpu().data.numpy()

                nosie_input_img_ed_low = pre.unitNormalisation(nosie_input_img_ed_low)
                nosie_input_img_es_low = pre.unitNormalisation(nosie_input_img_es_low)

                target_img_ed_low = SingleInference2d(E,D,nosie_input_img_ed_low,typemodel)
                target_img_es_low = SingleInference2d(E,D,nosie_input_img_es_low,typemodel)
            
            else:
                pad_input_img_ed_low = pre.pad_seq(input_img_ed_low)
                pad_input_img_es_low = pre.pad_seq(input_img_es_low)

                target_img_ed_low = SingleInference3D(Es,Et,Gs,Gt,Ns,Nt,pad_input_img_ed_low,typemodel)
                target_img_es_low = SingleInference3D(Es,Et,Gs,Gt,Ns,Nt,pad_input_img_es_low,typemodel)

                pad_target_img_ed_low = pre.pad_seq(target_img_ed_low)
                pad_target_img_es_low = pre.pad_seq(target_img_es_low)


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

    str_save = "... Total patients " + str(len(ed_mean_psnr)) + " Model Name " + typemodel      + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ED  "      + str(global_mean_endo_ed) + "/" + str(global_std_endo_ed) + "\n" + \
               " ... MYO ED  "       + str(global_mean_myo_ed)  + "/" + str(global_std_myo_ed)  + "\n" + \
               " ... RV ED  "        + str(global_mean_rv_ed)   + "/" + str(global_std_rv_ed)   + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ES  "      + str(global_mean_endo_es) + "/" + str(global_std_endo_es) + "\n" + \
               " ... MYO ES  "       + str(global_mean_myo_es)  + "/" + str(global_std_myo_es)  + "\n" + \
               " ... RV ES  "        + str(global_mean_rv_es)   + "/" + str(global_std_rv_es)   + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... PSNR ED  "      + str(ed_mean_psnr_global) + "/" + str(ed_std_psnr_global) + "\n" + \
               " ... SSIM ED  "      + str(ed_mean_ssim_global) + "/" + str(ed_std_ssim_global) + "\n" + \
               " ... PSNR ES  "      + str(es_mean_psnr_global) + "/" + str(es_std_psnr_global) + "\n" + \
               " ... SSIM ES  "      + str(es_mean_ssim_global) + "/" + str(es_std_ssim_global)
    text_file = open(os.path.join(pathSave,typemodel + "_table.txt"), "w")
    text_file.write(str_save)
    text_file.close()


if __name__ == '__main__':
    pathData = ""
    pathCode  = "" # Folder where is the list of patients (named dataset_txt)  
    pathsave  = "/mnt/sdb1/results/result_da"
    typedata  = "SG"
    pathModel = "/mnt/sdb1/models/IEEE_paper/SR_models/joint_unet_gan/generator.pt"
    typemodel = "V-AMA" #BicycleGAN, CycleGAN, MUNIT, V-AMA, no_dm (no domain adaption)
    n_worker  = 4
    style_dim = 8
    scanner_noise = False

    if typedata  == "SG":
        pathData = "" # Folder SG dataset
    else:
        pathData = "" # Folder UKBB dataset (only save - no GT) 
    

    today     =  date.today()
    pathSave  =  os.path.join(pathsave,"Test_DA_models_"+ str(today)) 
    mkdir_dir(pathSave)
    pathSave =  os.path.join(pathSave,typemodel) 
    mkdir_dir(pathSave)


    if   typemodel == "BicycleGAN":
            
            path_model_BicycleGAN = "" # path model
            G = model.Generator(100,100,z_dim=8)
            G.load_state_dict(torch.load(path_model_BicycleGAN))
            G.cuda().float()

            jointGenerator = load(pathModel)
                    
            testFunction(G,G,G,G,G,G,jointGenerator,pathCode,pathData,pathSave,typemodel,scanner_noise)

    elif typemodel == "CycleGAN":

            path_model_CycleGAN = "" # path model
            G = model.SimpleGenerator(100,100)
            G.load_state_dict(torch.load(path_model_CycleGAN))
            G.cuda().float()

            jointGenerator = load(pathModel)
            
            testFunction(G,G,G,G,G,G,jointGenerator,pathCode,pathData,pathSave,typemodel,scanner_noise)

    
    elif typemodel == "MUNIT":

            path_model_encoder_MUNIT = "./Encoder.pt" # path model
            path_model_decoder_MUNIT = "./Decoder.pt" # path model

            E = model.EncoderMUNIT(100,style_dim)
            E.load_state_dict(torch.load(path_model_encoder_MUNIT))
            E.cuda().float()

            D = model.DecoderMUNIT(100,style_dim)
            D.load_state_dict(torch.load(path_model_decoder_MUNIT))
            D.cuda().float()

            jointGenerator = load(pathModel)
            
            testFunction(E,D,D,D,D,D,jointGenerator,pathCode,pathData,pathSave,typemodel,scanner_noise)
    

    elif typemodel == "V-AMA":

            pathModel     = "" # Gemini-GAN path model
            ##############################################################################
            E_source_path = "./V-AMA/E_source.pt" # path model
            E_source   = model.EncoderVAE(100)
            E_source.load_state_dict(torch.load(E_source_path))
            Es   = E_source.float().cuda()

            E_target_path = "./V-AMA/E_target.pt" # path model
            E_target   = model.EncoderVAE(100)
            E_target.load_state_dict(torch.load(E_target_path))
            Et   = E_target.float().cuda()

            G_source_path = "./V-AMA/G_source.pt" # path model
            G_source   = model.DecoderBA(100)
            G_source.load_state_dict(torch.load(G_source_path))
            Gs   = G_source.float().cuda()

            Noise_source_path = "./V-AMA/Noise_source.pt" # path model
            Noise_source   = model.GetMeanStd(1)
            Noise_source.load_state_dict(torch.load(Noise_source_path))
            Ns   = Noise_source.float().cuda()

            Noise_target_path = "./V-AMA/Noise_target.pt"  # path model
            Noise_target   = model.GetMeanStd(1)
            Noise_target.load_state_dict(torch.load(Noise_target_path))
            Nt   = Noise_target.float().cuda()

            jointGenerator = load(pathModel)

            testFunction(Es,Et,Gs,Gs,Ns,Nt,jointGenerator,pathCode,pathData,pathSave,typemodel,scanner_noise)

    elif typemodel == "no_dm":
            jointGenerator = load(pathModel)
            testFunction_no_dm(jointGenerator,pathCode,pathData,pathSave,typemodel,scanner_noise)
