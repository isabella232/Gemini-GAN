
##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################


import warnings
from   multiprocessing import Pool
from   functools import partial
import os
from   tqdm  import tqdm
import shutil
import SimpleITK  as sitk
from   model import GeneratorUnetSegImg
import numpy as np 
import nibabel as nib
import cv2
import glob 
from   preprocessing import preprocessing
import torch
from   torch.autograd import Function, Variable
warnings.filterwarnings('ignore')

def preprocessing_ukbb(originalNii, data_dir):    
    os.system('cp '
              '{0} '
              '{1}/lvsa_.nii.gz'
              .format(originalNii, data_dir))

    os.system('autocontrast '
              '{0}/lvsa_.nii.gz '
              '{0}/lvsa_.nii.gz >/dev/nul '
              .format(data_dir))
        
    os.system('cardiacphasedetection '
              '{0}/lvsa_.nii.gz '
              '{0}/lvsa_ED.nii.gz '
              '{0}/lvsa_ES.nii.gz >/dev/nul '
              .format(data_dir))

    print('  Image preprocessing is done ...')

def get_preprocessing_ukbb(test_dir):
    for data in sorted(os.listdir(test_dir)):
            data_dir = os.path.join(test_dir, data)   
            #if os.path.exists('{0}/lvsa_.nii.gz'.format(data_dir)):
            #    os.system('rm {0}/lvsa_*.nii.gz'.format(data_dir))
            #    os.system('rm {0}/seg_*.nii.gz'.format(data_dir))
            
            originalnii = glob.glob('{0}/*.nii'.format(data_dir))     
            if not originalnii:
                print('  original nifit image does not exist, use lvsa.nii.gz')
                originalnii = glob.glob('{0}/*.nii.gz'.format(data_dir))  
                preprocessing_ukbb(originalnii[0], data_dir) 
            else:
                print('  start image preprocessing ...')
                preprocessing_ukbb(originalnii[0], data_dir)
            
            # Process ED and ES time frames
            image_ED_name = '{0}/lvsa_{1}.nii.gz'.format(data_dir, 'ED')
            image_ES_name = '{0}/lvsa_{1}.nii.gz'.format(data_dir, 'ES')

            if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                print(' Image {0} or {1} does not exist. Skip.'.format(image_ED_name, image_ES_name))
                continue

def load(path_model):
    generator     = GeneratorUnetSegImg(100,100).cuda()
    generator.load_state_dict(torch.load(path_model))
    generator.eval()
    generator     = generator.cuda()
    return generator

def get_torch(data):
    list_data = []
    for i in range(data.shape[0]): 
        list_data.append(data[i])
    torch_out = torch.from_numpy(np.asarray(list_data)).unsqueeze(0)   
    torch_out[torch_out != torch_out] = 0
    return torch_out

def remove_noise(data):
    filter_data = []
    kernel    = np.ones((3,3),np.float32)/9
    for i in range(data.shape[0]):
        dst       = cv2.filter2D(data[i],-1,kernel)
        filter_data.append(dst)
    return np.asarray(filter_data)

def fix_zhorizon(prediction,temp):
    zero   = np.zeros((temp.shape[1],temp.shape[2]))
    collect = []
    for i in range(temp.shape[0]):
        slices = temp [i]
        all_zeros = not slices.any()
        if all_zeros == True:
            collect.append(zero)
        else:
            collect.append(prediction[i])
    return np.asarray(collect)

def mul_list(new_a,new_b):
    return tuple([a*b for a,b in zip(new_a,new_b)])

def div_list(new_a,new_b):
    return tuple([a/b for a,b in zip(new_a,new_b)])

def GetNewSpacing(origin_size,new_size,origin_spacing):
    origin_size     = list(origin_size)
    new_size        = list(new_size)
    origin_spacing  = list(origin_spacing)
    new_spacing     = mul_list(origin_size, div_list(origin_spacing,new_size))
    return tuple(new_spacing)

def ResizeVolume(original_segmentation,size_data):
    new_spacing      = GetNewSpacing(original_segmentation.GetSize(),size_data,original_segmentation.GetSpacing())
    new_segmentation = sitk.Resample(original_segmentation, size_data,
                                    sitk.Transform(), 
                                    sitk.sitkBSpline,
                                    original_segmentation.GetOrigin(),
                                    new_spacing,
                                    original_segmentation.GetDirection(),
                                    0,
                                    original_segmentation.GetPixelID())
    return new_segmentation

def run_resample(original_path):
    for patient in os.listdir(original_path):
        path_seq      = os.path.join(original_path,patient,"enlarge_phases")
        path_seq_hr_res   = os.path.join(original_path,patient,"sequences_hr_resample")
        saveFolder(path_seq_hr_res)
        print("\n ... patient " +str(patient))
        for i in tqdm(range(len(os.listdir(path_seq)))):
            nii_volume      = os.path.join(path_seq,"lvsa_SR_"+ "{0:0=2d}".format(i) + ".nii.gz")
            sitk_volume     = sitk.ReadImage(nii_volume)
            np_volume       = sitk.GetArrayFromImage(sitk_volume)
            resample        = ResizeVolume(sitk_volume,np_volume.shape)
            nii_volume_save = os.path.join(path_seq_hr_res,"lvsa_SR_"+ "{0:0=2d}".format(i) + ".nii.gz")
            sitk.WriteImage(resample,nii_volume_save)

def write4Dresample(original_path):
    for patient in os.listdir(original_path):
        path_seq_lr         = os.path.join(original_path,patient,"enlarge_phases")
        save_path           = os.path.join(original_path,patient,"4D_rview")
        path_seq_hr_res     = os.path.join(original_path,patient,"sequences_hr_resample")
        list_images_lr,\
        list_images_hr      = [],[]
        for i in tqdm(range(len(os.listdir(path_seq_lr)))):
            nim_hr          = nib.load(os.path.join(path_seq_hr_res,"lvsa_SR_"+"{0:0=2d}".format(i)+".nii.gz"))
            image_hr        = nim_hr.get_data()
            image_hr        = np.squeeze(image_hr)
            list_images_hr += [image_hr]
        ### lr 
        lr_array      = np.array(list_images_lr,dtype=np.float64)
        lr_array      = np.transpose(lr_array, (1, 2, 3, 0))
        nim_lr_2      = nib.Nifti1Image(lr_array, nim_lr.affine, nim_lr.header)
        path_save_lr  = os.path.join(save_path,'4Dimg_HR_res.nii.gz')
        nib.save(nim_lr_2, path_save_lr)

def segPredict(pred_seg):
    _, pred_seg_max      = torch.max (pred_seg, dim=1)
    out = pred_seg_max.long().cpu().data.numpy()
    return out

def save_predictions(path_input,prediction_np,path_save):
    nim_input_LR_open  = nib.load(path_input)
    prediction_tr         = np.transpose(prediction_np, (2, 1, 0))
    nim_pred_SR_open   = nib.Nifti1Image(prediction_tr, nim_input_LR_open.affine, nim_input_LR_open.header)
    nib.save(nim_pred_SR_open,  path_save)


def run_SR_net(original_path,path_model,path_template):
    model = load(path_model)
    pre   = preprocessing(400,100)
    for patient in os.listdir(original_path):
        path_seq         = os.path.join(original_path,patient,"enlarge_phases")
        path_seq_hr_gray = os.path.join(original_path,patient,"sequences_HR_gray")
        path_seq_lr_gray = os.path.join(original_path,patient,"sequences_LR_gray")
        path_seq_hr_seg  = os.path.join(original_path,patient,"sequences_HR_seg")
        saveFolder(path_seq_hr_seg)
        saveFolder(path_seq_hr_gray)
        saveFolder(path_seq_lr_gray)
        print("\n ... GenScan patient " +str(patient))
        for i in tqdm(range(len(os.listdir(path_seq)))):
            templete_gray_HR    =  os.path.join(path_template,"grayscale_HR.nii.gz")
            templete_gray_LR    =  os.path.join(path_template,"grayscale_LR.nii.gz")
            templete_seg_HR     =  os.path.join(path_template,"segmentation_HR.nii.gz")
            ###############################################################
            sitk_templete_hr    = sitk.ReadImage(templete_gray_HR)
            temp_volume_hr      = sitk.GetArrayFromImage(sitk_templete_hr)
            ###############################################################
            sitk_templete_seg   = sitk.ReadImage(templete_seg_HR)
            temp_seg            = sitk.GetArrayFromImage(sitk_templete_seg)
            ###############################################################
            sitk_templete_lr    = sitk.ReadImage(templete_gray_LR)
            temp_volume_lr      = sitk.GetArrayFromImage(sitk_templete_lr)
            ###############################################################
            nii_volume          = os.path.join(path_seq,"lvsa_SR_"+ "{0:0=2d}".format(i) + ".nii.gz")
            sitk_volume         = sitk.ReadImage(nii_volume)
            np_volume           = sitk.GetArrayFromImage(sitk_volume)
            in_volume           = pre.pad_seq(np_volume)
            cuda_volume         = Variable(get_torch(in_volume).cuda().float())
            ###############################################################            
            hr_volume_gray,\
            hr_volume_seg   = model(cuda_volume)  #Gemini-GAN
            ###############################################################
            hr_volume_gray  = hr_volume_gray.cpu().data.numpy()[0]
            hr_volume_seg   = segPredict(hr_volume_seg)[0]
            ###############################################################
            hr_volume_gray_hr = pre.getRestoreImg(hr_volume_gray,temp_volume_hr)
            hr_volume_gray_lr = pre.getRestoreImg(np_volume,temp_volume_lr)
            hr_volume_seg     = pre.getRestoreImg(hr_volume_seg,temp_seg)
            ###############################################################
            nii_volume_gray_hr_save = os.path.join(path_seq_hr_gray,"lvsa_SR_"+ "{0:0=2d}".format(i) + ".nii.gz")
            nii_volume_seg_hr_save  = os.path.join(path_seq_hr_seg,"lvsa_SR_"+ "{0:0=2d}".format(i) + ".nii.gz")
            nii_volume_gray_lr_save  = os.path.join(path_seq_lr_gray,"lvsa_SR_"+ "{0:0=2d}".format(i) + ".nii.gz")
            ###############################################################
            save_predictions(templete_gray_HR,hr_volume_gray_hr,nii_volume_gray_hr_save)
            save_predictions(templete_gray_LR,hr_volume_gray_lr,nii_volume_gray_lr_save)
            save_predictions(templete_seg_HR,hr_volume_seg,nii_volume_seg_hr_save)
            ###############################################################

def preprocessing_cine(patient_save):
    phase_path_data = os.path.join(patient_save,"gray_phases")
    saveFolder(phase_path_data)
    print("\n ... Split sequence")

    os.system('splitvolume {0}/lvsa_.nii.gz {1}/lvsa_ -sequence'.format(patient_save,phase_path_data))

    phase_seg_path_data = os.path.join(patient_save,"motion")
    saveFolder(phase_seg_path_data)
    print("\n ... Resampling preprocessing generation")
    phase_path_data_resample = os.path.join(patient_save,"resample_phases")
    saveFolder(phase_path_data_resample)


    for fr in tqdm(range(len(os.listdir(phase_path_data)))):
        os.system('resample ' 
        '{0}/lvsa_{2}.nii.gz '
        '{1}/lvsa_{2}.nii.gz '
        '-size 1.25 1.25 2 >/dev/null'
        .format(phase_path_data,phase_path_data_resample, "{0:0=2d}".format(fr)))
    phase_path_data_enlarge = os.path.join(patient_save,"enlarge_phases")
    saveFolder(phase_path_data_enlarge) 


def apply(subj,data_dir):
    subject_dir = os.path.join(data_dir, subj)
    preprocessing_cine(model_dir,subject_dir)

def  preprocessingCine(patient_dir, coreNo):

    pool1 = Pool(processes = coreNo) 
    pool1.map(partial(apply, 
                     data_dir=patient_dir), 
                     sorted(os.listdir(dir_0)))   
    
    pool1.close() 
    pool1.join() 

def write4Dnii(original_path):
    for patient in os.listdir(original_path):
        save_path           = os.path.join(original_path,patient,"4D_rview")
        path_seq_hr_gray    = os.path.join(original_path,patient,"sequences_HR_gray")
        path_seq_lr_gray    = os.path.join(original_path,patient,"sequences_LR_gray")
        path_seq_hr_seg     = os.path.join(original_path,patient,"sequences_HR_seg")
        saveFolder(save_path)
        list_images_lr,\
        list_images_hr,\
        list_images_seg     = [],[],[]
        nim_lr              = None
        nim_hr              = None
        nim_seg             = None
        for i in tqdm(range(len(os.listdir(path_seq_hr_gray)))):
            nim_lr          = nib.load(os.path.join(path_seq_lr_gray,"lvsa_SR_"+"{0:0=2d}".format(i)+".nii.gz"))
            image_lr        = nim_lr.get_data()
            image_lr        = np.squeeze(image_lr)
            list_images_lr += [image_lr]
            nim_hr          = nib.load(os.path.join(path_seq_hr_gray,"lvsa_SR_"+"{0:0=2d}".format(i)+".nii.gz"))
            image_hr        = nim_hr.get_data()
            image_hr        = np.squeeze(image_hr)
            list_images_hr += [image_hr]
            nim_seg          = nib.load(os.path.join(path_seq_hr_seg,"lvsa_SR_"+"{0:0=2d}".format(i)+".nii.gz"))
            image_seg        = nim_seg.get_data()
            image_seg        = np.squeeze(image_seg)
            list_images_seg += [image_seg]
        ### lr 
        # batch * height * width * channels (=slices)
        lr_array                   = np.array(list_images_lr,dtype=np.float64)
        lr_array                   = np.transpose(lr_array, (1, 2, 3, 0))
        nim_lr_2                   = nib.Nifti1Image(lr_array, nim_lr.affine, nim_lr.header)
        path_save_lr               = os.path.join(save_path,'4Dimg_LR_GRAY.nii.gz')
        nib.save(nim_lr_2, path_save_lr)
        ### hr 
        hr_array                   = np.array(list_images_hr, dtype=np.float64)
        hr_array                   = np.transpose(hr_array, (1, 2, 3, 0))
        nim_hr_2                   = nib.Nifti1Image(hr_array, nim_hr.affine,nim_hr.header)
        path_save_hr               = os.path.join(save_path,'4Dimg_HR_GRAY.nii.gz') 
        nib.save(nim_hr_2,path_save_hr)
        ### seg
        seg_array                  = np.array(list_images_seg, dtype=np.float64)
        seg_array                  = np.transpose(seg_array, (1, 2, 3, 0))
        nim_seg_2                  = nib.Nifti1Image(seg_array, nim_seg.affine,nim_seg.header)
        path_save_seg              = os.path.join(save_path,'4Dimg_HR_SEG.nii.gz') 
        nib.save(nim_seg_2,path_save_seg)


if __name__ == '__main__':    
    open_path     =  "" # where is your patients folders
    coreNo        = 4 # number of cores for preprocessing
    path_template = ""  # where is your template volume
    save_path     = "" # folder where save 
    path_model    = "./generator.pt" # where is your Gemini-GAN model
    preprocessingCine(open_path,coreNo)
    run_SR_net(open_path,path_model,path_template)
    write4Dnii(open_path)



    


    
