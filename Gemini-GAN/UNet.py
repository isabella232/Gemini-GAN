##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################

from   torch.autograd import Function, Variable
import torch
from   model import GeneratorUnetSeg,GeneratorUnetImg
import numpy as np 
import torch.cuda as cutorch
import warnings
import torch.nn as     nn
from   dataset import TrainDataset
import torch
from   torch.utils.data import DataLoader
from   tqdm import tqdm
import os
import SimpleITK  as sitk
with   warnings.catch_warnings():
       warnings.filterwarnings("ignore")
from   metrics import * 
from   plots   import plots
import nibabel as nib
from   weight_initializer import Initializer
from   datetime import date
from   sklearn.metrics import f1_score
from   preprocessing import preprocessing


#########################################################
# Before starting the code please configure the paths   #
#########################################################



###########
# Setting #
####################################
# if want LMS support
lms_flag     = False
# if want GPU support 
cuda_flag    = True 
# debug save 
debug        = False
###################
beta_seg     = 1
# in channels 
in_channels  = 100 
# out_channels 
out_channels = 100
#################
every_valid  = 10 # every when (epoch) you want valid
################
plot  = plots()
################


##################
# Generatos type #
##################
name_G = "u_net_seg" #Please use u_net_seg/u_net_img -->  u_net_seg = only segmentation ,u_net_img =only super-resolution
##################


#####################
pathData       = "" # Folder where is LR/HR dataset 
#####################
pathsave       = "" # Folder where to save  
pathCode       = "" # Folder where is the list of patients (named dataset_txt)
####################


###########
# Dataset #
###########

n_worker             = 4 # number of CPU workers
batch_size           = 1 # batch size 
n_epochs             = 100 # number epochs   
device               = torch.device("cuda")
lr                   = 1e-3 # learning rate 
#############

generator = None

if name_G   ==   "u_net_seg":
    generator    = GeneratorUnetSeg   (in_channels,out_channels)
    generator    = generator.cuda()

elif name_G ==   "u_net_img":
    generator     = GeneratorUnetImg   (in_channels,out_channels)
    generator     = generator.cuda()


################
# Optimisation #
################

optimizer_G  = torch.optim.Adam(generator.parameters(),lr=lr, weight_decay=1e-6)
 
########
# Loss #
########

# Segmentation loss
generator_loss_seg  = nn.CrossEntropyLoss().to (device)
generator_loss_img  = nn.MSELoss().to (device)

#########
# utils #
#########


def open_txt(path):
    data = []
    with open(path, 'r') as f:
        data = [line.strip() for line in f]
    return data

def save_debug(save_path,train,valid):
    results_jpg = os.path.join(save_path,"unet_debug")
    mkdir_dir(results_jpg)
    train_path           = os.path.join(results_jpg,"train")
    valid_path           = os.path.join(results_jpg,"valid")
    mkdir_dir(train_path)
    mkdir_dir(valid_path)
    pre.seg(train,train_path)
    pre.seg(valid,valid_path)

def get_torch(data):
    data_torch  = torch.from_numpy(data)
    data_torch[data_torch != data_torch] = 0
    torch_out = data_torch.unsqueeze(0).float()
    return torch_out

def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

#################
# Create folder #
#################

today     =  date.today()
datatoday =  os.path.join(pathsave,"SR_results_"+ str(today)) 
mkdir_dir(datatoday)
pathsave  =  os.path.join(datatoday,name_G)
mkdir_dir(pathsave)


#########
# Valid #
#########

def calDice(seg,gt,k):
    return np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))

def checkNaN(val):
    p_val = 0
    if math.isnan(float(val)):
        p_val = 0
    else:
        p_val = val
    return p_val

def globaDice(seg,gt):
    endo_list,myo_list,rv_list = [],[],[]
    for i in range(seg.shape[0]):
        if np.mean(seg[i])>0 and np.mean(gt[i])>0:
            endo = calDice(seg[i],gt[i],1)
            myo  = calDice(seg[i],gt[i],2)
            rv   = calDice(seg[i],gt[i],3)
            if np.isnan(endo): continue 
            if np.isnan(myo):  continue 
            if np.isnan(rv):   continue 
            endo_list.append(endo)
            myo_list.append(myo)
            rv_list.append(rv)
    tot_dice = (checkNaN(np.mean(endo_list)) + checkNaN(np.mean(myo_list)) + checkNaN(np.mean(rv_list)))/3
    return tot_dice
        
def valid_seg(generator,path_valid,pathsave):

    generator.eval()
    psnr_val_ed     = []
    psnr_val_es     = []
    ssim_val_ed     = []
    ssim_val_es     = []
    dice_val_es     = []
    dice_val_ed     = []

    patients        = open_txt(os.path.join(pathCode,"dataset_txt","valid.txt"))
    pre             = preprocessing(400,out_channels)
    for i in tqdm(range(len(patients))):
        
        with torch.no_grad():

            ####################################################################################################
            save_path = os.path.join(pathsave,patients[i])
            ####################################################################################################
            path_input_sitk_ed_gs        = os.path.join(path_valid,patients[i],"low_gray_genscan_ED.nii.gz"  )
            path_ed_output_sitk_gs       = os.path.join(path_valid,patients[i],"hight_gray_genscan_ED.nii.gz")
            ####################################################################################################
            path_input_sitk_es_gs        = os.path.join(path_valid,patients[i],"low_gray_genscan_ES.nii.gz"  )
            path_es_output_sitk_gs       = os.path.join(path_valid,patients[i],"hight_gray_genscan_ES.nii.gz")
            ####################################################################################################
            path_input_sitk_ed_seg_gs    = os.path.join(path_valid,patients[i],"low_seg_genscan_ED.nii.gz"   )
            path_ed_output_sitk_seg_gs   = os.path.join(path_valid,patients[i],"hight_seg_genscan_ED.nii.gz" )
            #################################################################################################
            path_input_sitk_es_seg_gs    = os.path.join(path_valid,patients[i],"low_seg_genscan_ES.nii.gz"   )
            path_es_output_sitk_seg_gs   = os.path.join(path_valid,patients[i],"hight_seg_genscan_ES.nii.gz" )
            ################################################################################################
            input_img_ed_gs              = sitk.GetArrayFromImage(sitk.ReadImage(path_input_sitk_ed_gs))
            input_img_es_gs              = sitk.GetArrayFromImage(sitk.ReadImage(path_input_sitk_es_gs))
            ###################################################################
            input_img_ed_pre_gs          = pre.pad_seq(input_img_ed_gs)
            ###################################################################
            input_img_es_pre_gs          = pre.pad_seq(input_img_es_gs)
            ###################################################################
            input_img_ed_gs_cuda         = Variable(get_torch(input_img_ed_pre_gs).to(device).float())
            input_img_es_gs_cuda         = Variable(get_torch(input_img_es_pre_gs).to(device).float())
            ###################################################################
            hight_ed_pred_seg_gs         = generator(input_img_ed_gs_cuda)
            _, hight_ed_pred_seg_gs      = torch.max (hight_ed_pred_seg_gs, dim=1)
            hight_ed_pred_seg_gs         = hight_ed_pred_seg_gs.cpu().data.numpy()
            ####################################################################
            hight_es_pred_seg_gs         = generator(input_img_es_gs_cuda)
            _, hight_es_pred_seg_gs      = torch.max (hight_es_pred_seg_gs, dim=1)
            hight_es_pred_seg_gs         = hight_es_pred_seg_gs.cpu().data.numpy()
            ####################################################################
            path_pred_gray_genscan_ed    = os.path.join(save_path,"pred_hight_gray_genscan_ED.nii.gz")
            path_pred_gray_genscan_es    = os.path.join(save_path,"pred_hight_gray_genscan_ES.nii.gz")
            ######################################################################
            hight_ed_pred_gs_seg         = pre.getRestoreImg(hight_ed_pred_seg_gs [0],input_img_ed_gs)
            hight_es_pred_gs_seg         = pre.getRestoreImg(hight_es_pred_seg_gs [0],input_img_es_gs)
            ######################################################################

            ###########
            # Metric  #
            ###########

            gt_hight_seg_genscan_ES     = sitk.GetArrayFromImage(sitk.ReadImage(path_es_output_sitk_seg_gs))
            gt_hight_seg_genscan_ES[gt_hight_seg_genscan_ES[:,:,:]==4]=3
            gt_hight_seg_genscan_ED     = sitk.GetArrayFromImage(sitk.ReadImage(path_ed_output_sitk_seg_gs))
            gt_hight_seg_genscan_ED[gt_hight_seg_genscan_ED[:,:,:]==4]=3
            dice_ES = globaDice(hight_es_pred_gs_seg,gt_hight_seg_genscan_ES)
            dice_ED = globaDice(hight_ed_pred_gs_seg,gt_hight_seg_genscan_ED)
            dice_val_es.append(dice_ES)
            dice_val_ed.append(dice_ED)
            

    final_dice_es     = np.mean(dice_val_es)
    final_dice_ed     = np.mean(dice_val_ed)
    return  final_dice_es,final_dice_ed


def valid_img(generator,path_valid,pathsave):
    generator.eval()
    psnr_val_ed     = []
    psnr_val_es     = []
    ssim_val_ed     = []
    ssim_val_es     = []
    dice_val_es     = []
    dice_val_ed     = []
    patients        = open_txt(os.path.join(pathCode,"dataset_txt","valid.txt"))
    pre             = preprocessing(400,out_channels)
    for i in tqdm(range(len(patients))):
        with torch.no_grad():
            save_path = os.path.join(pathsave,patients[i])
            ####################################################################################################
            path_input_sitk_ed_gs        = os.path.join(path_valid,patients[i],"low_gray_genscan_ED.nii.gz"  )
            path_ed_output_sitk_gs       = os.path.join(path_valid,patients[i],"hight_gray_genscan_ED.nii.gz")
            ####################################################################################################
            path_input_sitk_es_gs        = os.path.join(path_valid,patients[i],"low_gray_genscan_ES.nii.gz"  )
            path_es_output_sitk_gs       = os.path.join(path_valid,patients[i],"hight_gray_genscan_ES.nii.gz")
            ####################################################################################################
            path_input_sitk_ed_seg_gs    = os.path.join(path_valid,patients[i],"low_seg_genscan_ED.nii.gz"   )
            path_ed_output_sitk_seg_gs   = os.path.join(path_valid,patients[i],"hight_seg_genscan_ED.nii.gz" )
            #################################################################################################
            path_input_sitk_es_seg_gs    = os.path.join(path_valid,patients[i],"low_seg_genscan_ES.nii.gz"   )
            path_es_output_sitk_seg_gs   = os.path.join(path_valid,patients[i],"hight_seg_genscan_ES.nii.gz" )
            ################################################################################################
            input_img_ed_gs              = sitk.GetArrayFromImage(sitk.ReadImage(path_input_sitk_ed_gs))
            input_img_es_gs              = sitk.GetArrayFromImage(sitk.ReadImage(path_input_sitk_es_gs))
            ###################################################################
            input_img_ed_pre_gs          = pre.pad_seq(input_img_ed_gs)
            ###################################################################
            input_img_es_pre_gs          = pre.pad_seq(input_img_es_gs)
            ###################################################################
            input_img_ed_gs_cuda         = Variable(get_torch(input_img_ed_pre_gs).to(device).float())
            input_img_es_gs_cuda         = Variable(get_torch(input_img_es_pre_gs).to(device).float())
            ###################################################################
            hight_ed_pred_img_gs         = generator(input_img_ed_gs_cuda)
            ####################################################################
            hight_es_pred_img_gs         = generator(input_img_es_gs_cuda)
            ####################################################################
            path_pred_gray_genscan_ed    = os.path.join(save_path,"pred_hight_gray_genscan_ED.nii.gz")
            path_pred_gray_genscan_es    = os.path.join(save_path,"pred_hight_gray_genscan_ES.nii.gz")
            ######################################################################
            hight_ed_pred_gs             = hight_ed_pred_img_gs.cpu().data.numpy()
            ####################################################################
            hight_es_pred_gs             = hight_es_pred_img_gs.cpu().data.numpy()
            ####################################################################
            hight_ed_pred_gs             = pre.getRestoreImg(hight_ed_pred_gs [0],input_img_ed_gs)
            hight_es_pred_gs             = pre.getRestoreImg(hight_es_pred_gs [0],input_img_es_gs)
            ######################################################################

        
            ###########
            # Metric  #
            ###########

            psnr_val_ed.append(getPSNR(hight_ed_pred_gs,sitk.GetArrayFromImage(sitk.ReadImage(path_ed_output_sitk_gs))))
            psnr_val_es.append(getPSNR(hight_es_pred_gs,sitk.GetArrayFromImage(sitk.ReadImage(path_es_output_sitk_gs))))
            ssim_val_ed.append(ssim(pre.padImg(hight_ed_pred_gs), pre.padImg(sitk.GetArrayFromImage(sitk.ReadImage(path_ed_output_sitk_gs)))))
            ssim_val_es.append(ssim(pre.padImg(hight_es_pred_gs), pre.padImg(sitk.GetArrayFromImage(sitk.ReadImage(path_es_output_sitk_gs)))))
            
    final_psnr_val_ed = np.mean(psnr_val_ed)
    final_psnr_val_es = np.mean(psnr_val_es)
    final_ssim_val_ed = np.mean(ssim_val_ed)
    final_ssim_val_es = np.mean(ssim_val_es)
    return  final_psnr_val_ed,final_psnr_val_es,\
            final_ssim_val_ed,final_ssim_val_es


############
# Training #
############

def trainSeg(generator,input_lr,input_hr_seg):
    generator.zero_grad()
    output_seg_hr       = generator(input_lr)
    g_loss              = generator_loss_seg(output_seg_hr,input_hr_seg.unsqueeze(0)) 
    g_loss.backward()
    optimizer_G.step()
    return generator,g_loss,output_seg_hr

def trainImgSeg(generator,input_lr,input_hr,input_hr_seg):
    generator.zero_grad()
    output_hr,\
    output_seg_hr   = generator(input_lr)
    g_loss          = generator_loss_img(output_hr[0],input_hr[0]) + generator_loss_seg(output_seg_hr,input_hr_seg.unsqueeze(0)) 
    g_loss.backward()
    optimizer_G.step()
    return generator,g_loss,output_hr,output_seg_hr


def trainImg(generator,input_lr,input_hr):
    generator.zero_grad()
    output_hr       = generator(input_lr)
    g_loss          = generator_loss_img(output_hr[0],input_hr[0]) 
    g_loss.backward()
    optimizer_G.step()
    return generator,g_loss,output_hr


valid_set             = TrainDataset(pathData,pathCode,out_channels)
training_data_loader  = DataLoader(dataset=valid_set, num_workers=n_worker, batch_size=batch_size)
#######################
global_dice_es     = []
global_dice_ed     = []
global_psnr_val_ed = []
global_psnr_val_es = []
global_ssim_val_ed = []
global_ssim_val_es = []
#######################
pre = preprocessing(400,out_channels)
######################
for epoch in range(n_epochs):
    print("\n ... Epoch " + str(epoch))

    generator.train()    
    for batch_idx, batch_train in tqdm(enumerate(training_data_loader,\
                                    1),total=len(training_data_loader)):
        
        ######################################
        # Retrive numpy data from dataLoder  #
        ######################################

        np_input_model,\
        np_gt_model,np_seg                                = batch_train
        np_input_model[np_input_model != np_input_model]  = 0
        np_gt_model[np_gt_model       != np_gt_model]     = 0
        input_img                                         = Variable(np_input_model.to(device)).float()
        input_gt                                          = Variable(np_gt_model.to(device)).float()
        gt_seg                                            = Variable(np_seg.to(device).squeeze(0).long())

        ####################
        # Backpropagation  #
        ####################
        
        if name_G == "u_net_seg":
            generator,g_loss,seg = trainSeg(generator,input_img,gt_seg)
        else:
            generator,g_loss,img = trainImg(generator,input_img,input_gt)
    
    
    if epoch > 0 and epoch % every_valid==0:
    
        ###################################################################################
        if name_G == "u_net_seg":
            epoch_save  = os.path.join(pathsave,"Epoch_"+str(epoch))
            mkdir_dir(epoch_save)
            print("\n ... Valid")
            ##################################################
            final_dice_es,final_dice_ed = valid_seg(generator,pathData,epoch_save)
            ##################################################
            global_dice_es.append(final_dice_es)
            global_dice_ed.append(final_dice_ed)
            torch.save(generator.state_dict(),os.path.join(epoch_save,"generator.pt"))
            plot.dice_plot(global_dice_es,epoch_save,"ES")
            plot.dice_plot(global_dice_ed,epoch_save,"ED")

        elif name_G == "u_net_img":
            epoch_save  = os.path.join(pathsave,"Epoch_"+str(epoch))
            mkdir_dir(epoch_save)
            print("\n ... Valid")
            ##################################################
            final_psnr_val_ed,final_psnr_val_es,\
            final_ssim_val_ed,final_ssim_val_es  = valid_img(generator,pathData,epoch_save)
            ##################################################
            global_psnr_val_ed.append(final_psnr_val_ed)
            global_psnr_val_es.append(final_psnr_val_es)
            global_ssim_val_ed.append(final_ssim_val_ed)
            global_ssim_val_es.append(final_ssim_val_es)
            plot.psnr_plot(global_psnr_val_ed,epoch_save,"ED")
            plot.psnr_plot(global_psnr_val_es,epoch_save,"ES")
            plot.ssim_plot(global_ssim_val_ed,epoch_save,"ED")
            plot.ssim_plot(global_ssim_val_es,epoch_save,"ES")
            torch.save(generator.state_dict(),os.path.join(epoch_save,"generator.pt"))
        ###################################################################################






