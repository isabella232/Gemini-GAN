############################################
# Nicolo Savioli, Imperial Collage London  # 
############################################

from   torch.autograd    import Function, Variable
import torch
from   model import GeneratorSRGANSegImg,GeneratorSRGANSeg,GeneratorSRGANImg,\
                    Discriminator,DiscriminatorOriginal,GeneratorSRGANSegImgDouble
import numpy as np 
import torch.cuda as cutorch
import warnings
import torch.nn as     nn
from   dataset import TrainDataset
import torch
from   torch.utils.data import DataLoader
from   tqdm import tqdm
import cv2,os
import SimpleITK  as sitk
with   warnings.catch_warnings():
       warnings.filterwarnings("ignore")
from   metrics import * 
from   plots   import plots
from   preprocessing import preprocessing
import nibabel as nib
from weight_initializer import Initializer
from datetime import date
from sklearn.metrics import f1_score
from pretrain import *


#########################################################
# Before starting the code please configure the paths   #
#########################################################

###########
# Setting #
###########

###################
beta_seg     = 1
# in channels 
in_channels  = 100 
# out_channels 
out_channels = 100
###################

################################
# Paths for Dataset            #
################################

#####################
pathData       = "" # Folder where is LR/HR dataset 
#####################
pathsave       = "" # Folder where to save  
pathCode       = "" # Folder where is the list of patients (named dataset_txt)
####################

###################
flag_GAN       = True  # GAN=True if GAN, False if don't want it. 
###################

##################
# Generatos type #
##################
name_G = "SR" 
##################

##################
type_net = "img" # Please use joint/seg/img --> (joint = seg_SR_GAN, img = super-resolution SR_GAN, seg = segmentation SR_GAN)
##################

##################
if flag_GAN:
    name_G = type_net +"_"+ name_G + "_GAN"
else:
    name_G = type_net +"_"+ name_G 
##################


###########
# Dataset #
###########

####################################
n_worker             = 4 # number of CPU workers
batch_size           = 1 # Batch size 
n_epochs             = 100 # Number epochs  
device               = torch.device("cuda")
every_valid          = 10 #  When (epoch) the model is valid
lr                   = 1e-3 # learning rate 
####################################

####################################
plot   = plots()
####################################
    
#############
# get model #
#############

def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
########
# Load #
########

if type_net =="seg":
    generator = GeneratorSRGANSeg(in_channels,out_channels,4)
    generator = generator.cuda()
    netD = DiscriminatorOriginal(in_channels,1)
    netD = netD.cuda()  
elif type_net =="img":
    generator = GeneratorSRGANImg(in_channels,out_channels,4)
    generator = generator.cuda()
    netD = DiscriminatorOriginal(in_channels,1)
    netD = netD.cuda()  
elif type_net =="joint":
    generator = GeneratorSRGANSegImg(in_channels,out_channels,4)
    generator = generator.cuda()
    netD = DiscriminatorOriginal(in_channels,2)
    netD = netD.cuda()  


today     =  date.today()
datatoday =  os.path.join(pathsave,"SR_results_"+ str(today)) 
mkdir_dir(datatoday)
pathsave  =  os.path.join(datatoday,name_G)
mkdir_dir(pathsave)

################
# Optimisation #
################

optimizer_G  = torch.optim.Adam(generator.parameters(),lr=lr, weight_decay=1e-6)
optimizer_D  = torch.optim.Adam(netD.parameters(),lr=lr, weight_decay=1e-6)

########
# Loss #
#######


# Segmentation loss
generator_loss_seg  = nn.CrossEntropyLoss().to (device)
generator_loss_img  = nn.MSELoss().to (device)
        
def open_txt(path):
    data = []
    with open(path, 'r') as f:
        data = [line.strip() for line in f]
    return data


###############
# Valid utils #
###############

def get_torch(data):
    torch_out = torch.from_numpy(data).unsqueeze(0).to(device)
    torch_out[torch_out != torch_out] = 0
    return Variable(torch_out)

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
        
def valid_seg_img(generator,path_valid,pathsave):
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
            hight_ed_pred_img_gs,\
            hight_ed_pred_seg_gs         = generator(input_img_ed_gs_cuda)
            _, hight_ed_pred_seg_gs      = torch.max (hight_ed_pred_seg_gs, dim=1)
            hight_ed_pred_seg_gs         = hight_ed_pred_seg_gs.cpu().data.numpy()
            ####################################################################
            hight_es_pred_img_gs,\
            hight_es_pred_seg_gs         = generator(input_img_es_gs_cuda)
            _, hight_es_pred_seg_gs      = torch.max (hight_es_pred_seg_gs, dim=1)
            hight_es_pred_seg_gs         = hight_es_pred_seg_gs.cpu().data.numpy()
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
            hight_ed_pred_gs_seg         = pre.getRestoreImg(hight_ed_pred_seg_gs [0],input_img_ed_gs)
            hight_es_pred_gs_seg         = pre.getRestoreImg(hight_es_pred_seg_gs [0],input_img_es_gs)
            ######################################################################
            ###########
            # Metric  #
            ###########

            ###
            gt_hight_seg_genscan_ES     = sitk.GetArrayFromImage(sitk.ReadImage(path_es_output_sitk_seg_gs))
            gt_hight_seg_genscan_ES[gt_hight_seg_genscan_ES[:,:,:]==4]=3
            gt_hight_seg_genscan_ED     = sitk.GetArrayFromImage(sitk.ReadImage(path_ed_output_sitk_seg_gs))
            gt_hight_seg_genscan_ED[gt_hight_seg_genscan_ED[:,:,:]==4]=3
            dice_ES = globaDice(hight_es_pred_gs_seg,gt_hight_seg_genscan_ES)
            dice_ED = globaDice(hight_ed_pred_gs_seg,gt_hight_seg_genscan_ED)
            ###
            dice_val_es.append(dice_ES)
            dice_val_ed.append(dice_ED)
            ###
            psnr_val_ed.append(getPSNR(hight_ed_pred_gs,sitk.GetArrayFromImage(sitk.ReadImage(path_ed_output_sitk_gs))))
            psnr_val_es.append(getPSNR(hight_es_pred_gs,sitk.GetArrayFromImage(sitk.ReadImage(path_es_output_sitk_gs))))
            ssim_val_ed.append(ssim(pre.padImg(hight_ed_pred_gs), pre.padImg(sitk.GetArrayFromImage(sitk.ReadImage(path_ed_output_sitk_gs)))))
            ssim_val_es.append(ssim(pre.padImg(hight_es_pred_gs), pre.padImg(sitk.GetArrayFromImage(sitk.ReadImage(path_es_output_sitk_gs)))))
            ###

    final_dice_es     = np.mean(dice_val_es)
    final_dice_ed     = np.mean(dice_val_ed)
    final_psnr_val_ed = np.mean(psnr_val_ed)
    final_psnr_val_es = np.mean(psnr_val_es)
    final_ssim_val_ed = np.mean(ssim_val_ed)
    final_ssim_val_es = np.mean(ssim_val_es)
    return  final_dice_es,final_dice_ed,\
            final_psnr_val_ed,final_psnr_val_es,\
            final_ssim_val_ed,final_ssim_val_es



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
            hight_es_pred_img_gs         = generator(input_img_es_gs_cuda)
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

            ###
            psnr_val_ed.append(getPSNR(hight_ed_pred_gs,sitk.GetArrayFromImage(sitk.ReadImage(path_ed_output_sitk_gs))))
            psnr_val_es.append(getPSNR(hight_es_pred_gs,sitk.GetArrayFromImage(sitk.ReadImage(path_es_output_sitk_gs))))
            ssim_val_ed.append(ssim(pre.padImg(hight_ed_pred_gs), pre.padImg(sitk.GetArrayFromImage(sitk.ReadImage(path_ed_output_sitk_gs)))))
            ssim_val_es.append(ssim(pre.padImg(hight_es_pred_gs), pre.padImg(sitk.GetArrayFromImage(sitk.ReadImage(path_es_output_sitk_gs)))))
            ###

    final_psnr_val_ed = np.mean(psnr_val_ed)
    final_psnr_val_es = np.mean(psnr_val_es)
    final_ssim_val_ed = np.mean(ssim_val_ed)
    final_ssim_val_es = np.mean(ssim_val_es)

    return  final_psnr_val_ed,final_psnr_val_es,\
            final_ssim_val_ed,final_ssim_val_es


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
            ####################################################################
            _, hight_es_pred_seg_gs      = torch.max (hight_es_pred_seg_gs, dim=1)
            hight_es_pred_seg_gs         = hight_es_pred_seg_gs.cpu().data.numpy()
            ####################################################################
            path_pred_gray_genscan_ed    = os.path.join(save_path,"pred_hight_gray_genscan_ED.nii.gz")
            path_pred_gray_genscan_es    = os.path.join(save_path,"pred_hight_gray_genscan_ES.nii.gz")
            ######################################################################

            ######################################################################
            hight_ed_pred_gs_seg         = pre.getRestoreImg(hight_ed_pred_seg_gs [0],input_img_ed_gs)
            hight_es_pred_gs_seg         = pre.getRestoreImg(hight_es_pred_seg_gs [0],input_img_es_gs)
            ######################################################################

            ###########
            # Metric  #
            ###########

            ###
            gt_hight_seg_genscan_ES     = sitk.GetArrayFromImage(sitk.ReadImage(path_es_output_sitk_seg_gs))
            gt_hight_seg_genscan_ES[gt_hight_seg_genscan_ES[:,:,:]==4]=3
            gt_hight_seg_genscan_ED     = sitk.GetArrayFromImage(sitk.ReadImage(path_ed_output_sitk_seg_gs))
            gt_hight_seg_genscan_ED[gt_hight_seg_genscan_ED[:,:,:]==4]=3
            dice_ES = globaDice(hight_es_pred_gs_seg,gt_hight_seg_genscan_ES)
            dice_ED = globaDice(hight_ed_pred_gs_seg,gt_hight_seg_genscan_ED)
            ###
            dice_val_es.append(dice_ES)
            dice_val_ed.append(dice_ED)
            ###

    final_dice_es     = np.mean(dice_val_es)
    final_dice_ed     = np.mean(dice_val_ed)

    return  final_dice_es,final_dice_ed


############
# Training #
############

def train(generator,input_lr,input_hr,input_hr_seg):

    if type_net == "joint":
        generator.zero_grad()
        ######################################
        output_hr,\
        output_seg_hr   = generator(input_lr)
        #####################################
        g_loss          = generator_loss_img(output_hr[0],input_hr[0]) + generator_loss_seg(output_seg_hr,input_hr_seg.unsqueeze(0)) 
        g_loss.backward()
        optimizer_G.step()
        return generator

    elif type_net == "seg":
        generator.zero_grad()
        ######################################
        output_seg_hr   = generator(input_lr)
        #####################################
        g_loss          = generator_loss_seg(output_seg_hr,input_hr_seg.unsqueeze(0)) 
        g_loss.backward()
        optimizer_G.step()
        return generator
    
    elif type_net == "img":
        generator.zero_grad()
        ######################################
        output_hr = generator(input_lr)
        #####################################
        g_loss          = generator_loss_img(output_hr[0],input_hr[0]) 
        g_loss.backward()
        optimizer_G.step()
        return generator



def trainGAN(generator,input_lr,input_hr,input_hr_seg):

    valid = Variable(torch.cuda.FloatTensor (1, 1).fill_(1.0), requires_grad=False)
    fake = Variable(torch.cuda.FloatTensor (1, 1).fill_(0.0), requires_grad=False)

    # -----------------
    #  Train Generator
    # -----------------

    generator.zero_grad()


    if type_net == "joint" or type_net == "joint_double":
        # Generate a batch of images
        output_hr,\
        output_seg_hr  = generator(input_lr)
    
        # Loss measures generator's ability to fool the discriminator
        _, max_output_seg_hr  = torch.max (output_seg_hr, dim=1)
        max_output_seg_hr = max_output_seg_hr.float()
        input_fake_D = torch.cat ((output_hr,max_output_seg_hr),0)
        input_fake_D = input_fake_D.view((input_fake_D.size(0)*input_fake_D.size(1),input_fake_D.size(2),input_fake_D.size(3))).unsqueeze(0)
        g_loss = generator_loss_img(output_hr[0],input_hr[0]) + generator_loss_seg(output_seg_hr,input_hr_seg.unsqueeze(0)) + 1e-3*generator_loss_img(netD(input_fake_D), valid)
    
    elif type_net == "seg":
        output_seg_hr  = generator(input_lr)
        # Loss measures generator's ability to fool the discriminator
        _, max_output_seg_hr  = torch.max (output_seg_hr, dim=1)
        max_output_seg_hr = max_output_seg_hr.float()
        input_fake_D = max_output_seg_hr
        input_fake_D = input_fake_D.view((input_fake_D.size(0)*input_fake_D.size(1),input_fake_D.size(2),input_fake_D.size(3))).unsqueeze(0)
        g_loss =  generator_loss_seg(output_seg_hr,input_hr_seg.unsqueeze(0)) + 1e-3*generator_loss_img(netD(input_fake_D), valid)
    
    elif type_net == "img":
        # Generate a batch of images
        output_hr = generator(input_lr)
        input_fake_D = output_hr
        input_fake_D = input_fake_D.view((input_fake_D.size(0)*input_fake_D.size(1),input_fake_D.size(2),input_fake_D.size(3))).unsqueeze(0)
        g_loss = generator_loss_img(output_hr[0],input_hr[0]) + 1e-3*generator_loss_img(netD(input_fake_D), valid)
     
    
    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    if type_net == "joint" or type_net == "joint_double":
        input_hr_seg_f = input_hr_seg.float()
        input_real_D = torch.cat((input_hr[0],input_hr_seg_f),0).unsqueeze(0)
        real_loss = generator_loss_img(netD(input_real_D), valid)
        fake_loss = generator_loss_img(netD(input_fake_D.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
    elif type_net == "seg":
        input_hr_seg_f = input_hr_seg.float()
        input_real_D = input_hr_seg_f.unsqueeze(0)
        real_loss = generator_loss_img(netD(input_real_D), valid)
        fake_loss = generator_loss_img(netD(input_fake_D.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
    elif type_net == "img":
        input_hr_seg_f = input_hr_seg.float()
        input_real_D = input_hr[0].unsqueeze(0)
        real_loss = generator_loss_img(netD(input_real_D), valid)
        fake_loss = generator_loss_img(netD(input_fake_D.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
    return generator


valid_set = TrainDataset(pathData,pathCode,out_channels)
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
    # train mode
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
        
        if flag_GAN:
            netG  = trainGAN(generator,input_img,input_gt,gt_seg)
        else:
            netG  = train(generator,input_img,input_gt,gt_seg)


    ###############################################################################   
    if epoch > 0 and epoch % every_valid==0:

        if type_net == "joint" or type_net == "joint_double": 
            print("\n ... Valid")
            epoch_save  = os.path.join(pathsave,"Epoch_"+str(epoch))
            mkdir_dir(epoch_save)
            final_dice_es,final_dice_ed,\
            final_psnr_val_ed,final_psnr_val_es,\
            final_ssim_val_ed,final_ssim_val_es = valid_seg_img(generator,pathData,epoch_save)
            global_psnr_val_ed.append(final_psnr_val_ed)
            global_psnr_val_es.append(final_psnr_val_es)
            global_ssim_val_ed.append(final_ssim_val_ed)
            global_ssim_val_es.append(final_ssim_val_es)
            global_dice_es.append(final_dice_es)
            global_dice_ed.append(final_dice_ed)
            plot.dice_plot(global_dice_es,epoch_save,"ES")
            plot.dice_plot(global_dice_ed,epoch_save,"ED")
            plot.psnr_plot(global_psnr_val_ed,epoch_save,"ED")
            plot.psnr_plot(global_psnr_val_es,epoch_save,"ES")
            plot.ssim_plot(global_ssim_val_ed,epoch_save,"ED")
            plot.ssim_plot(global_ssim_val_es,epoch_save,"ES")
            torch.save(generator.state_dict(),os.path.join(epoch_save,"generator.pt"))

        elif type_net == "img":
            print("\n ... Valid")
            epoch_save  = os.path.join(pathsave,"Epoch_"+str(epoch))
            mkdir_dir(epoch_save)
            final_psnr_val_ed,final_psnr_val_es,\
            final_ssim_val_ed,final_ssim_val_es = valid_img(generator,pathData,epoch_save)
            global_psnr_val_ed.append(final_psnr_val_ed)
            global_psnr_val_es.append(final_psnr_val_es)
            global_ssim_val_ed.append(final_ssim_val_ed)
            global_ssim_val_es.append(final_ssim_val_es)
            plot.dice_plot(global_dice_es,epoch_save,"ES")
            plot.dice_plot(global_dice_ed,epoch_save,"ED")
            plot.psnr_plot(global_psnr_val_ed,epoch_save,"ED")
            plot.psnr_plot(global_psnr_val_es,epoch_save,"ES")
            plot.ssim_plot(global_ssim_val_ed,epoch_save,"ED")
            plot.ssim_plot(global_ssim_val_es,epoch_save,"ES")
            torch.save(generator.state_dict(),os.path.join(epoch_save,"generator.pt"))

        elif type_net == "seg":
            print("\n ... Valid")
            epoch_save  = os.path.join(pathsave,"Epoch_"+str(epoch))
            mkdir_dir(epoch_save)
            final_dice_es,final_dice_ed = valid_seg(generator,pathData,epoch_save)
            global_dice_es.append(final_dice_es)
            global_dice_ed.append(final_dice_ed)
            plot.dice_plot(global_dice_es,epoch_save,"ES")
            plot.dice_plot(global_dice_ed,epoch_save,"ED")
            plot.psnr_plot(global_psnr_val_ed,epoch_save,"ED")
            plot.psnr_plot(global_psnr_val_es,epoch_save,"ES")
            plot.ssim_plot(global_ssim_val_ed,epoch_save,"ED")
            plot.ssim_plot(global_ssim_val_es,epoch_save,"ES")
            torch.save(generator.state_dict(),os.path.join(epoch_save,"generator.pt"))

    ###############################################################################



