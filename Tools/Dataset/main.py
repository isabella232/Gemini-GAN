##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################

import os 
from   shutil import copyfile
import SimpleITK  as sitk
import numpy as np 
from   operator import truediv
from   random import randint
import secrets 
import numpy as np, nibabel as nib
import pickle
import csv
import pandas as pd  
import shutil
from   os import path
import pickle


def mkdir_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def nii_load_itk(path):
    return sitk.ReadImage(path)

def convert_to_numpy(Image):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def load_vol(path):
     return convert_to_numpy(nii_load_itk(path))

def setDirections(high_gray,low_gray,high_seg,low_seg):
    low_gray.SetDirection(high_gray.GetDirection())
    high_seg.SetDirection(high_gray.GetDirection())
    low_seg.SetDirection (high_gray.GetDirection())
    return high_gray,low_gray,high_seg,low_seg

def reshape(vol,outsize):
    return  nn.functional.interpolate(vol,size=outsize,mode='trilinear',align_corners=True)


def mappingSegBack_ED(data_dir):
    nim     = nib.load('{0}/lvsa_ED.nii.gz'.format(data_dir))
    lvsa_ED = nim.get_data()
    lvsa_ED = np.squeeze(lvsa_ED, axis=-1).astype(np.int16)

    lvsa_ED_cropped = nib.load('{0}/lvsa_ED_cropped.nii.gz'.format(data_dir)).get_data()
    lvsa_ED_cropped = np.squeeze(lvsa_ED_cropped, axis=-1).astype(np.int16)
            
    lvsa_ED_enlarged = nib.load('{0}/lvsa_ED_enlarged.nii.gz'.format(data_dir)).get_data()
    lvsa_ED_enlarged = np.squeeze(lvsa_ED_enlarged, axis=-1).astype(np.int16)

    seg_lvsa_ED_enlarged = nib.load('{0}/seg_lvsa_ED_enlarged.nii.gz'.format(data_dir)).get_data()
    seg_lvsa_ED_enlarged = np.squeeze(seg_lvsa_ED_enlarged, axis=-1).astype(np.int16)

    st = int((lvsa_ED_enlarged.shape[2]-lvsa_ED_cropped.shape[2])/2)
    ed = lvsa_ED_enlarged.shape[2] - st 
    seg_lvsa_ED_cropped = seg_lvsa_ED_enlarged[:,:,st:ed]

    orgM, orgN, orgZ = lvsa_ED.shape[:3]
    cropM, cropN = lvsa_ED_cropped.shape[:2]

    seg_lvsa_ED = np.zeros((orgM, orgN, orgZ), dtype=np.int16)

    for i in range(orgM-cropM):
        for j in range(orgN-cropN):
           
            if np.array_equal(lvsa_ED[i:i+cropM,j:j+cropN,:], lvsa_ED_cropped):
                seg_lvsa_ED[i:i+cropM,j:j+cropN,:] = seg_lvsa_ED_cropped

    nim2 = nib.Nifti1Image(seg_lvsa_ED, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/seg_lvsa_ED_high_res.nii.gz'.format(data_dir))


def mappingSegBack_ES(data_dir): 
    nim = nib.load('{0}/lvsa_ES.nii.gz'.format(data_dir))
    lvsa_ES = nim.get_data()
    lvsa_ES = np.squeeze(lvsa_ES, axis=-1).astype(np.int16)

    lvsa_ES_cropped = nib.load('{0}/lvsa_ES_cropped.nii.gz'.format(data_dir)).get_data()
    lvsa_ES_cropped = np.squeeze(lvsa_ES_cropped, axis=-1).astype(np.int16)
            
    lvsa_ES_enlarged = nib.load('{0}/lvsa_ES_enlarged.nii.gz'.format(data_dir)).get_data()
    lvsa_ES_enlarged = np.squeeze(lvsa_ES_enlarged, axis=-1).astype(np.int16)

    seg_lvsa_ES_enlarged = nib.load('{0}/seg_lvsa_ES_enlarged.nii.gz'.format(data_dir)).get_data()
    seg_lvsa_ES_enlarged = np.squeeze(seg_lvsa_ES_enlarged, axis=-1).astype(np.int16)

    st = int((lvsa_ES_enlarged.shape[2]-lvsa_ES_cropped.shape[2])/2)
    ed = lvsa_ES_enlarged.shape[2] - st 
    seg_lvsa_ES_cropped = seg_lvsa_ES_enlarged[:,:,st:ed]

    orgM, orgN, orgZ = lvsa_ES.shape[:3]
    cropM, cropN = lvsa_ES_cropped.shape[:2]

    seg_lvsa_ES = np.zeros((orgM, orgN, orgZ), dtype=np.int16)

    for i in range(orgM-cropM):
        for j in range(orgN-cropN):
            
            if np.array_equal(lvsa_ES[i:i+cropM,j:j+cropN,:], lvsa_ES_cropped):
                seg_lvsa_ES[i:i+cropM,j:j+cropN,:] = seg_lvsa_ES_cropped

    nim2 = nib.Nifti1Image(seg_lvsa_ES, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/seg_lvsa_ES_high_res.nii.gz'.format(data_dir))

def converting(data_dir):
    for fr in ['ED', 'ES']: 
        image_name_in = '{0}/lvsa_{1}.gipl'.format(data_dir, fr)
        image_name_out = '{0}/lvsa_{1}.nii.gz'.format(data_dir, fr)
        
        label_name_in = '{0}/segmentation_{1}.gipl'.format(data_dir, fr)
        label_name_out = '{0}/seg_lvsa_{1}_enlarged.nii.gz'.format(data_dir, fr)
        
        if os.path.exists(image_name_in):
            os.system('convert {0} {1}'.format(image_name_in,image_name_out))
            os.system('convert {0} {1}'.format(label_name_in,label_name_out))

def FixFiles(new_path,patient):
    datapath    = os.path.join(new_path,patient)
    high_ED     = sitk.ReadImage(os.path.join(datapath,"genscan_high_grayscale",   "lvsa_ED.gipl"         ))       
    high_ES     = sitk.ReadImage(os.path.join(datapath,"genscan_high_grayscale",   "lvsa_ES.gipl"         )) 
    low_ED      = sitk.ReadImage(os.path.join(datapath,"genscan_low_grayscale",    "lvsa_SR_ED.nii.gz"    ))    
    low_ES      = sitk.ReadImage(os.path.join(datapath,"genscan_low_grayscale",    "lvsa_SR_ES.nii.gz"    )) 
    low_seg_ED  = sitk.ReadImage(os.path.join(datapath,"genscan_low_segmentation", "LVSA_seg_ED.nii.gz"   ))    
    low_seg_ES  = sitk.ReadImage(os.path.join(datapath,"genscan_low_segmentation", "LVSA_seg_ES.nii.gz"   )) 
    high_seg_ED = sitk.ReadImage(os.path.join(datapath,"genscan_high_segmentation","seg_lvsa_ED_high_res.nii.gz"))    
    high_seg_ES = sitk.ReadImage(os.path.join(datapath,"genscan_high_segmentation","seg_lvsa_ES_high_res.nii.gz"))
    #######################################
    low_ED.SetDirection        (high_ED.GetDirection())
    low_ED.SetOrigin           (high_ED.GetOrigin())
    low_ED.SetSpacing          (high_ED.GetSpacing())
    #######################################
    low_ES.SetDirection        (high_ED.GetDirection())
    low_ES.SetOrigin           (high_ED.GetOrigin())
    low_ES.SetSpacing          (high_ED.GetSpacing())
    #######################################
    low_seg_ED.SetDirection    (high_ED.GetDirection())
    low_seg_ED.SetOrigin       (high_ED.GetOrigin())
    low_seg_ED.SetSpacing      (high_ED.GetSpacing())
    #######################################
    low_seg_ES.SetDirection    (high_ED.GetDirection())
    low_seg_ES.SetOrigin       (high_ED.GetOrigin())
    low_seg_ES.SetSpacing      (high_ED.GetSpacing())
    #######################################
    high_seg_ED.SetDirection   (high_ED.GetDirection())
    high_seg_ED.SetOrigin      (high_ED.GetOrigin())
    high_seg_ED.SetSpacing     (high_ED.GetSpacing())
    ########################################
    high_seg_ES.SetDirection   (high_ED.GetDirection())
    high_seg_ES.SetOrigin      (high_ED.GetOrigin())
    high_seg_ES.SetSpacing     (high_ED.GetSpacing())
    #################################################################################
    sitk.WriteImage(high_ED,os.path.join(datapath,"genscan_high_grayscale","lvsa_ED.gipl"))
    sitk.WriteImage(high_ES,os.path.join(datapath,"genscan_high_grayscale","lvsa_ES.gipl"))
    #################################################################################
    sitk.WriteImage(low_ED,os.path.join(datapath,"genscan_low_grayscale","lvsa_SR_ED.nii.gz"))
    sitk.WriteImage(low_ES,os.path.join(datapath,"genscan_low_grayscale","lvsa_SR_ES.nii.gz"))
    #################################################################################  
    sitk.WriteImage(low_seg_ED,os.path.join(datapath,"genscan_low_segmentation","LVSA_seg_ED.nii.gz"))
    sitk.WriteImage(low_seg_ES,os.path.join(datapath,"genscan_low_segmentation","LVSA_seg_ES.nii.gz"))
    #################################################################################
    sitk.WriteImage(high_seg_ED,os.path.join(datapath,"genscan_high_segmentation","seg_lvsa_ED_high_res.nii.gz"))
    sitk.WriteImage(high_seg_ES,os.path.join(datapath,"genscan_high_segmentation","seg_lvsa_ES_high_res.nii.gz"))
    #################################################################################

def get_new_folders(new_path,patientname):
    mkdir_dir(os.path.join(new_path,patientname))
    high_path                 = os.path.join(new_path,patientname,"genscan_high_grayscale")
    low_path                  = os.path.join(new_path,patientname,"genscan_low_grayscale")
    syn_path                  = os.path.join(new_path,patientname,"genscan_syn_grayscale")
    high_seg1                 = os.path.join(new_path,patientname,"genscan_high_segmentation_for_high")
    high_seg2                 = os.path.join(new_path,patientname,"genscan_high_segmentation_for_low")
    low_ukbb_grayscale_high   = os.path.join(new_path,patientname,"ukbb_syn_grayscale")
    low_ukbb_grayscale_low    = os.path.join(new_path,patientname,"ukbb_low_grayscale")
    high_seg_ukbb_seg_high    = os.path.join(new_path,patientname,"ukbb_high_segmentation")
    low_seg_ukbb_seg_low      = os.path.join(new_path,patientname,"ukbb_low_segmentation")
    not_resampling            = os.path.join(new_path,patientname,"final_data")
    genscan_low_seg           = os.path.join(new_path,patientname,"genscan_low_segmentation")
    mkdir_dir (high_path)
    mkdir_dir (low_path)
    mkdir_dir (high_seg1)
    mkdir_dir (high_seg2)
    mkdir_dir (low_ukbb_grayscale_high)
    mkdir_dir (high_seg_ukbb_seg_high)
    mkdir_dir (low_seg_ukbb_seg_low)
    mkdir_dir (not_resampling)
    mkdir_dir (low_ukbb_grayscale_low)
    mkdir_dir (syn_path)
    mkdir_dir(genscan_low_seg)
    return high_path,low_path,\
           high_seg1,high_seg2,\
           low_ukbb_grayscale_high,\
           high_seg_ukbb_seg_high,\
           low_seg_ukbb_seg_low,\
           not_resampling,\
           low_ukbb_grayscale_low,\
           syn_path,\
           genscan_low_seg


def scan_high_paths(high_path,target):
    list_high_res_paths = os.listdir(high_path)
    get_path            = ""
    flag                = False
    for path in list_high_res_paths:
        if target in path.split("_"):
           get_path = path
           flag     = True 
           break
    return  get_path,flag


def getVariableData(x):
    device    = torch.device("cpu")
    return Variable(torch.from_numpy(x).to(device,\
                            dtype=torch.float32).unsqueeze_(0).unsqueeze_(1))     

def volumetricResize(vol,sizeVol):
    resize_vol  = nn.functional.interpolate(getVariableData(vol),size=sizeVol,\
                                                mode='trilinear',align_corners=True).numpy()[0][0]
    return sitk.GetImageFromArray(resize_vol)


def resizeVol(listPath,sizeVol):
    list_size = []
    size_list = []
    for le in listPath:
        lvol = sitk.GetArrayFromImage(sitk.ReadImage(le))
        list_size.append(int(lvol.shape[0]))
    maxdata = max(list_size)
    for le in listPath: 
        lvol    =  sitk.GetArrayFromImage(sitk.ReadImage(le))


        resize_vol  = nn.functional.interpolate(getVariableData(vol),size=(maxdata,sizeVol,sizeVol),\
                                                mode='trilinear',align_corners=True).numpy()[0][0]
        save_vol    = sitk.GetImageFromArray(resize_vol)
        sitk.WriteImage(save_vol,le)

def clean_data(pathData):
     tot_cont = 0
     dealte_cont = 0 
     for patient in os.listdir(pathData):
          print("\n ..." + patient)
          high_grayscale = os.path.join(pathData,patient,"high_grayscale")
          low_grayscale  = os.path.join(pathData,patient,"low_grayscale")
          h_dirContents  = os.listdir  (high_grayscale)
          l_dirContents  = os.listdir  (low_grayscale)
          if len(h_dirContents) == 0 or len(l_dirContents) == 0:
               print("\n ... remove: " + os.path.join(pathData,patient))
               shutil.rmtree(os.path.join(pathData,patient))
               dealte_cont += 1 
          tot_cont += 1
     tot = tot_cont - dealte_cont
     print("\n\n\n ... number of patients are: " + str(tot))


def div_list(new_a,new_b):
    return tuple([a/b for a,b in zip(new_a,new_b)])

def mul_list(new_a,new_b):
    return tuple([a*b for a,b in zip(new_a,new_b)])

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
                                    sitk.sitkNearestNeighbor,
                                    original_segmentation.GetOrigin(),
                                    new_spacing,
                                    original_segmentation.GetDirection(),
                                    0,
                                    original_segmentation.GetPixelID())
    return new_segmentation


def checkFolder(UKBBPath,low_files_ukbb_grayscale_high,\
                low_files_ukbb_grayscale_low,\
                seg_files_ukbb_high,seg_file_ukbb_low,len_size):
    list_patient_ukbb = os.listdir(UKBBPath)
    correct_folders   = []
    for data in list_patient_ukbb:
        flag_check1 = flag_check2 = flag_check3 = flag_check4 = True
        for fr in low_files_ukbb_grayscale_high:
            if path.exists(os.path.join(UKBBPath,data,fr)) == False:
                flag_check1 = False
        for fr in low_files_ukbb_grayscale_low:
            if path.exists(os.path.join(UKBBPath,data,fr)) == False:
                flag_check2 = False
        for fr in seg_files_ukbb_high:
            if path.exists(os.path.join(UKBBPath,data,fr)) == False:
                flag_check3 = False
        for fr in seg_file_ukbb_low:
            if path.exists(os.path.join(UKBBPath,data,fr)) == False:
                flag_check4 = False

        if flag_check1 and flag_check2 and flag_check3 and flag_check4:
            print("\n ... UKBB Patient "+ data +" correct")
            correct_folders.append(data)
        else:
            print("\n ... UKBB Patient "+ data +" NOT correct") 

        if int(len(correct_folders)) > int(len(len_size)):
           break
    return correct_folders


def getRandomUKBB(GenScanPath,UKBBPath,low_files_ukbb_grayscale_high,\
                  low_files_ukbb_grayscale_low,seg_files_ukbb_high,\
                  seg_file_ukbb_low):
    list_patient_genscan          = os.listdir(GenScanPath)
    
    #list_patient_ukbb             = checkFolder(UKBBPath,low_files_ukbb_grayscale_high,\
    #                                            low_files_ukbb_grayscale_low,seg_files_ukbb_high,\
    #                                            seg_file_ukbb_low,list_patient_genscan)
    #with open('./cardiac/4D_superesolution_data/mean_std_data/ukbb_list', 'wb') as fp:
    #    pickle.dump(list_patient_ukbb, fp)
    with open('./cardiac/4D_superesolution_data/mean_std_data/ukbb_list', 'rb') as fp:
        list_patient_ukbb =  pickle.load(fp)



    print("\n ... Number of UKBB correct_folders are: " + str(len(list_patient_ukbb)))
    secure_random                 = secrets.SystemRandom()
    list_of_random_ukbb_patients  = secure_random.sample(list_patient_ukbb, len(list_patient_genscan))
    return list_of_random_ukbb_patients


def saveInfo(info,pathdata):
    file = open(os.path.join(pathdata,"info.txt"), "w") 
    file.write("UKBB patient name:" + info) 
    file.close() 

def MeanStd(low_path,high_path,\
           ukbb_path,new_path,low_files,\
           high_files,seg_high1,\
           seg_high2,\
           size_high,low_files_ukbb_grayscale_high,\
           low_files_ukbb_grayscale_low,\
           seg_files_ukbb_high,seg_file_ukbb_low,size_low):

    ##################################
    low_files_List                = []
    high_files_List               = []
    ##################################
    low_ukbb_grayscale_List_high  = []
    low_ukbb_grayscale_List_low   = []
    ##################################
    global_mean = []
    global_std  = []
    ################
    #cont = 0       #
    ################
    cont_patients = 0 
    #################

    ukbb_patients = getRandomUKBB(low_path,ukbb_path)

    with open(os.path.join(new_path,"ukbb_patients"), 'wb') as fp:
        pickle.dump(ukbb_patients, fp)

    for patient in os.listdir(low_path):    
        old_high_path,flag = scan_high_paths(high_path,patient)
        if os.path.isdir(os.path.join(high_path,old_high_path)):
            if flag==True:

                print("\n ... " + patient )

                ################
                # GenScan data #
                ################

                for fr in low_files:
                    low_files_List.append(os.path.join(low_path,patient,fr))
                for fr in high_files:
                    high_files_List.append(os.path.join(high_path,old_high_path,fr))

                #############
                # ukbb data #
                #############

                for fr in low_files_ukbb_grayscale_high: 
                    low_ukbb_grayscale_List_high.append(os.path.join(ukbb_path,ukbb_patients[cont_patients],fr))

                for fr in low_files_ukbb_grayscale_low: 
                    low_ukbb_grayscale_List_low.append(os.path.join(ukbb_path,ukbb_patients[cont_patients],fr))

                ##############################################################
                np_grayscale_ED_high            = sitk.GetArrayFromImage(sitk.ReadImage(high_files_List[0]))
                np_grayscale_ES_high            = sitk.GetArrayFromImage(sitk.ReadImage(high_files_List[1]))
                ##############################################################
                np_grayscale_ED_low             = sitk.GetArrayFromImage(sitk.ReadImage(low_files_List[0]))
                np_grayscale_ES_low             = sitk.GetArrayFromImage(sitk.ReadImage(low_files_List[1]))
                ##############################################################  
                np_low_ukbb_grayscale_ED_high   = sitk.GetArrayFromImage(sitk.ReadImage(low_ukbb_grayscale_List_high[0]))
                np_low_ukbb_grayscale_ES_high   = sitk.GetArrayFromImage(sitk.ReadImage(low_ukbb_grayscale_List_high[1]))
                ##############################################################
                np_low_ukbb_grayscale_ED_low    = sitk.GetArrayFromImage(sitk.ReadImage(low_ukbb_grayscale_List_low[0]))
                np_low_ukbb_grayscale_ES_low    = sitk.GetArrayFromImage(sitk.ReadImage(low_ukbb_grayscale_List_low[1]))
                ##############################################################

                ########
                # Mean #
                ########

                ##############################################################
                np_grayscale_ED_high_mean            = np.mean(np_grayscale_ED_high)
                np_grayscale_ES_high_mean            = np.mean(np_grayscale_ES_high)
                ##############################################################
                np_grayscale_ED_low_mean             = np.mean(np_grayscale_ED_low)
                np_grayscale_ES_low_mean             = np.mean(np_grayscale_ES_low)
                ##############################################################  
                np_low_ukbb_grayscale_ED_high_mean   = np.mean(np_low_ukbb_grayscale_ED_high)
                np_low_ukbb_grayscale_ES_high_mean   = np.mean(np_low_ukbb_grayscale_ES_high)
                ##############################################################
                np_low_ukbb_grayscale_ED_low_mean    = np.mean(np_low_ukbb_grayscale_ED_low)
                np_low_ukbb_grayscale_ES_low_mean    = np.mean(np_low_ukbb_grayscale_ES_low)
                ##############################################################
                global_mean.append(np_grayscale_ED_high_mean)
                global_mean.append(np_grayscale_ES_high_mean)
                global_mean.append(np_grayscale_ED_low_mean)
                global_mean.append(np_grayscale_ES_low_mean)
                global_mean.append(np_low_ukbb_grayscale_ED_high_mean)
                global_mean.append(np_low_ukbb_grayscale_ES_high_mean)
                global_mean.append(np_low_ukbb_grayscale_ED_low_mean)
                global_mean.append(np_low_ukbb_grayscale_ES_low_mean)
                ##############################################################

                ########
                # Std #
                #######

                ##############################################################
                np_grayscale_ED_high_std            = np.std(np_grayscale_ED_high)
                np_grayscale_ES_high_std            = np.std(np_grayscale_ES_high)
                ##############################################################
                np_grayscale_ED_low_std             = np.std(np_grayscale_ED_low)
                np_grayscale_ES_low_std             = np.std(np_grayscale_ES_low)
                ##############################################################  
                np_low_ukbb_grayscale_ED_high_std   = np.std(np_low_ukbb_grayscale_ED_high)
                np_low_ukbb_grayscale_ES_high_std   = np.std(np_low_ukbb_grayscale_ES_high)
                ##############################################################
                np_low_ukbb_grayscale_ED_low_std    = np.std(np_low_ukbb_grayscale_ED_low)
                np_low_ukbb_grayscale_ES_low_std    = np.std(np_low_ukbb_grayscale_ES_low)
                ##############################################################
                global_std.append(np_grayscale_ED_high_std)
                global_std.append(np_grayscale_ES_high_std)
                global_std.append(np_grayscale_ED_low_std)
                global_std.append(np_grayscale_ES_low_std)
                global_std.append(np_low_ukbb_grayscale_ED_high_std)
                global_std.append(np_low_ukbb_grayscale_ES_high_std)
                global_std.append(np_low_ukbb_grayscale_ED_low_std)
                global_std.append(np_low_ukbb_grayscale_ES_low_std)
                ##############################################################

                cont_patients += 1

                #if cont > 1:
                #    break
                #cont += 1

    mean_data = np.mean(global_mean)
    std_data  = np.std(global_std)

    with open(os.path.join(new_path,"mean_std.csv"), mode='w') as meanstdfile:
        writer = csv.writer(meanstdfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(mean_data),str(std_data)])


def normalized(v):
        v_min = np.amin(v)
        v_max = np.amax(v)
        out   = (v - v_min)/(v_max - v_min)
        return out 




def normalisation(img,mean,std):
    new_img = sitk.GetImageFromArray(normalized(sitk.GetArrayFromImage(img)))
    new_img.CopyInformation(img)
    return  new_img

def get_mean_std(path):
    data = pd.read_csv(path)  
    return float(data.columns[0]),float(data.columns[1])

def FixHeader_tmp(temp,new): 
    new.CopyInformation(temp)
    return new


def FixHeader(new,tmp):
    #new.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
    new.SetDirection(tmp.GetDirection())
    new.SetOrigin(tmp.GetOrigin())
    #new.SetSpacing(new.GetSpacing())
    return new


def getnii(low_path,high_path,\
           ukbb_path,new_path,low_files_syn,\
           low_files_low,high_files,seg_high1,\
           seg_high2,\
           size_high,low_files_ukbb_grayscale_high,\
           low_files_ukbb_grayscale_low,\
           seg_files_ukbb_high,seg_file_ukbb_low,size_low,\
           path_open,low_files_seg):
    list_not      = []   
    cont_patients = 0 
    ##################
    # open ukbb   #
    ##################
    with open (os.path.join(path_open,"ukbb_patients"), 'rb') as fp:
        #ukbb_patients = pickle.load(fp)
        ukbb_patients  = getRandomUKBB(low_path,ukbb_path,low_files_ukbb_grayscale_high,\
                                       low_files_ukbb_grayscale_low,seg_files_ukbb_high,\
                                       seg_file_ukbb_low)
    ##############################################
    mean,std =  get_mean_std(os.path.join(path_open,"mean_std.csv"))
    #print (" \n ... mean: " + str(mean) + " std: " + str(std) + " number of patients: " + str(len(ukbb_patients)))
    #############
    cont = 0
    for patient in os.listdir(low_path):
        #try:
            old_high_path,flag = scan_high_paths(high_path,patient)
            if os.path.isdir(os.path.join(high_path,old_high_path)):
                if os.path.isdir(os.path.join(ukbb_path,ukbb_patients[cont_patients])):
                    if flag==True:
                        print("\n ... GenScan " + patient + " - UKBB "+ ukbb_patients[cont_patients])
                        # convert files in high path folder 
                        converting(os.path.join(high_path,old_high_path))
                        # mapping ED Segmentation back to original image
                        mappingSegBack_ED(os.path.join(high_path,old_high_path))
                        # mapping ES Segmentation back to original image
                        mappingSegBack_ES(os.path.join(high_path,old_high_path))
                        ####################################################
                        low_files_syn_List            = []
                        low_files_low_List            = []
                        high_files_List               = []
                        seg_high_List                 = []
                        seg_low_List                  = []
                        low_files_seg_List            = []
                        ##################################
                        low_ukbb_grayscale_List_high  = []
                        low_ukbb_grayscale_List_low   = []
                        ##################################
                        high_seg_ukbb_high_List       = []
                        low_seg_ukbb_seg_low_List     = []
                        #################################################### 
                        high_new_path,\
                        low_new_path,\
                        high_seg1_save,\
                        high_seg2_save,\
                        low_ukbb_grayscale_high,\
                        high_seg_ukbb_high,\
                        low_seg_ukbb_seg_low,\
                        not_resampling,\
                        low_ukbb_grayscale_low,\
                        syn_new_path,low_path_segmentation = get_new_folders(new_path,patient)
                        #####################################################
                        ################
                        # GenScan data #
                        ################

                        for fr in low_files_syn:
                            copyfile(os.path.join(low_path,patient,fr), os.path.join(syn_new_path,fr))
                            low_files_syn_List.append(os.path.join(syn_new_path,fr))
                        for fr in low_files_low:
                            copyfile(os.path.join(low_path,patient,fr), os.path.join(low_new_path,fr))
                            low_files_low_List.append(os.path.join(low_new_path,fr))
                        for fr in high_files:
                            copyfile(os.path.join(high_path,old_high_path,fr), os.path.join(high_new_path,fr))
                            high_files_List.append(os.path.join(high_new_path,fr))
                        for fr in seg_high1:
                            copyfile(os.path.join(high_path,old_high_path,fr), os.path.join(high_seg1_save,fr))
                            seg_high_List.append(os.path.join(high_seg1_save,fr))
                        for fr in seg_high2:
                            copyfile(os.path.join(low_path,patient,fr), os.path.join(high_seg2_save,fr))
                            seg_low_List.append(os.path.join(high_seg2_save,fr))
                        for fr in low_files_seg:
                            copyfile(os.path.join(low_path,patient,fr), os.path.join(low_path_segmentation,fr))
                            low_files_seg_List.append(os.path.join(low_path_segmentation,fr))
    
                        #############
                        # ukbb data #
                        #############

                        for fr in low_files_ukbb_grayscale_high: 
                            copyfile(os.path.join(ukbb_path,ukbb_patients[cont_patients],fr), os.path.join(low_ukbb_grayscale_high,fr))
                            low_ukbb_grayscale_List_high.append(os.path.join(low_ukbb_grayscale_high,fr))

                        for fr in low_files_ukbb_grayscale_low: 
                            copyfile(os.path.join(ukbb_path,ukbb_patients[cont_patients],fr), os.path.join(low_ukbb_grayscale_low,fr))
                            low_ukbb_grayscale_List_low.append(os.path.join(low_ukbb_grayscale_low,fr))

                        for fr in seg_files_ukbb_high: 
                            copyfile(os.path.join(ukbb_path,ukbb_patients[cont_patients],fr), os.path.join(high_seg_ukbb_high,fr))
                            high_seg_ukbb_high_List.append(os.path.join(high_seg_ukbb_high,fr))
                        
                        for fr in seg_file_ukbb_low: 
                            copyfile(os.path.join(ukbb_path,ukbb_patients[cont_patients],fr), os.path.join(low_seg_ukbb_seg_low,fr))
                            low_seg_ukbb_seg_low_List.append(os.path.join(low_seg_ukbb_seg_low,fr))
                            
                        ##############################################################
                        saveInfo(ukbb_patients[cont_patients],low_ukbb_grayscale_high)
                        saveInfo(ukbb_patients[cont_patients],high_seg_ukbb_high)
                        saveInfo(ukbb_patients[cont_patients],low_seg_ukbb_seg_low)
                        ##############################################################

                        ###########
                        # GenScan #
                        ###########

                        ##############################################################
                        sitk_grayscale_ED_high    = normalisation(sitk.ReadImage(high_files_List[0]),mean,std)
                        sitk_grayscale_ES_high    = normalisation(sitk.ReadImage(high_files_List[1]),mean,std)
                        ##############################################################
                        sitk_grayscale_ED_low     = normalisation(sitk.ReadImage(low_files_low_List[0]),mean,std)
                        sitk_grayscale_ES_low     = normalisation(sitk.ReadImage(low_files_low_List[1]),mean,std)
                        ##############################################################
                        sitk_grayscale_ED_syn     = normalisation(sitk.ReadImage(low_files_syn_List[0]),mean,std)
                        sitk_grayscale_ES_syn     = normalisation(sitk.ReadImage(low_files_syn_List[1]),mean,std)
                        ##############################################################
                        sitk_segmentation_ED_high = sitk.ReadImage(seg_high_List[0])
                        sitk_segmentation_ES_high = sitk.ReadImage(seg_high_List[1])
                        ##############################################################
                        sitk_segmentation_ED_low  = sitk.ReadImage(seg_low_List[0])
                        sitk_segmentation_ES_low  = sitk.ReadImage(seg_low_List[1])
                        ##############################################################
                        sitk_segmentation_LOW_ED  = sitk.ReadImage(low_files_seg_List[0])
                        sitk_segmentation_LOW_ES  = sitk.ReadImage(low_files_seg_List[1])
                        ##############################################################
                        
                        ###########
                        # UKBB    #
                        ###########
                        
                        ##############################################################
                        sitk_low_ukbb_grayscale_ED_high    = normalisation(sitk.ReadImage(low_ukbb_grayscale_List_high[0]),mean,std)
                        sitk_low_ukbb_grayscale_ES_high    = normalisation(sitk.ReadImage(low_ukbb_grayscale_List_high[1]),mean,std)
                        ##############################################################
                        sitk_low_ukbb_grayscale_ED_low     = normalisation(sitk.ReadImage(low_ukbb_grayscale_List_low[0]),mean,std)
                        sitk_low_ukbb_grayscale_ES_low     = normalisation(sitk.ReadImage(low_ukbb_grayscale_List_low[1]),mean,std)
                        ##############################################################
                        sitk_high_seg_ukbb_high_ED         = sitk.ReadImage(high_seg_ukbb_high_List[0])
                        sitk_high_seg_ukbb_high_ES         = sitk.ReadImage(high_seg_ukbb_high_List[1])
                        ##############################################################
                        sitk_low_seg_ukbb_seg_low_ED       = sitk.ReadImage(low_seg_ukbb_seg_low_List[0])
                        sitk_low_seg_ukbb_seg_low_ES       = sitk.ReadImage(low_seg_ukbb_seg_low_List[1])
                        ##############################################################
                        
                        ################
                        # Resampling   #
                        ################
                        sitk_grayscale_ED_high_resampling_resize            = ResizeVolume(sitk_segmentation_ED_low,sitk_segmentation_LOW_ED.GetSize())
                        sitk_grayscale_ES_high_resampling_high_resize       = ResizeVolume(sitk_segmentation_ES_low,sitk_segmentation_LOW_ES.GetSize())
                        
                        sitk_grayscale_ED_high_resampling_resize_ukbb       = ResizeVolume(sitk_high_seg_ukbb_high_ED,sitk_low_seg_ukbb_seg_low_ED.GetSize())
                        sitk_grayscale_ES_high_resampling_high_resize_ukbb  = ResizeVolume(sitk_high_seg_ukbb_high_ES,sitk_low_seg_ukbb_seg_low_ES.GetSize())
                        
                        #########################
                        # Save not resize data  #
                        #########################
                        sitk.WriteImage(sitk_grayscale_ED_low,os.path.join(not_resampling,"GENSCAN_NO_SYN_ED.nii.gz"))
                        sitk.WriteImage(sitk_grayscale_ES_low,os.path.join(not_resampling,"GENSCAN_NO_SYN_ES.nii.gz"))
                        ########################
                        sitk.WriteImage(sitk_segmentation_ED_low,os.path.join(not_resampling,"GENSCAN_HIGH_SEG_ED.nii.gz"))
                        sitk.WriteImage(sitk_segmentation_ES_low,os.path.join(not_resampling,"GENSCAN_HIGH_SEG_ES.nii.gz"))
                        ########################
                        sitk.WriteImage(sitk_grayscale_ED_syn,os.path.join(not_resampling,"GENSCAN_SYN_ED.nii.gz"))
                        sitk.WriteImage(sitk_grayscale_ES_syn,os.path.join(not_resampling,"GENSCAN_SYN_ES.nii.gz"))
                        ########################
                        sitk.WriteImage(sitk_segmentation_LOW_ED,os.path.join(not_resampling,"GENSCAN_LOW_SEG_ED.nii.gz"))
                        sitk.WriteImage(sitk_segmentation_LOW_ES,os.path.join(not_resampling,"GENSCAN_LOW_SEG_ES.nii.gz"))
                        ########################
                        sitk.WriteImage(sitk_grayscale_ED_high_resampling_resize,os.path.join(not_resampling,"GENSCAN_HIGH_RESAMPLING_SEG_ED.nii.gz"))
                        sitk.WriteImage(sitk_grayscale_ES_high_resampling_high_resize,os.path.join(not_resampling,"GENSCAN_HIGH_RESAMPLING_SEG_ES.nii.gz"))
                        ########################
                        sitk.WriteImage(sitk_grayscale_ED_high,os.path.join(not_resampling,"GENSCAN_HIGH_RESOLUTION_ED.nii.gz"))
                        sitk.WriteImage(sitk_grayscale_ES_high,os.path.join(not_resampling,"GENSCAN_HIGH_RESOLUTION_ES.nii.gz"))
                        ########################
                        sitk.WriteImage(sitk_low_ukbb_grayscale_ED_high,os.path.join(not_resampling,"UKBB_SYN_ED.nii.gz"))
                        sitk.WriteImage(sitk_low_ukbb_grayscale_ES_high,os.path.join(not_resampling,"UKBB_SYN_ES.nii.gz"))
                        ########################
                        sitk.WriteImage(sitk_low_ukbb_grayscale_ED_low,os.path.join(not_resampling,"UKBB_NO_SYN_ED.nii.gz"))
                        sitk.WriteImage(sitk_low_ukbb_grayscale_ES_low,os.path.join(not_resampling,"UKBB_NO_SYN_ES.nii.gz"))  
                        ########################
                        sitk.WriteImage(sitk_high_seg_ukbb_high_ED,os.path.join(not_resampling,"UKBB_HIGH_SEG_ED.nii.gz"))
                        sitk.WriteImage(sitk_high_seg_ukbb_high_ES,os.path.join(not_resampling,"UKBB_HIGH_SEG_ES.nii.gz"))
                        ########################
                        sitk.WriteImage(sitk_low_seg_ukbb_seg_low_ED,os.path.join(not_resampling,"UKBB_LOW_SEG_ED.nii.gz"))
                        sitk.WriteImage(sitk_low_seg_ukbb_seg_low_ES,os.path.join(not_resampling,"UKBB_LOW_SEG_ES.nii.gz"))
                        ########################
                        sitk.WriteImage(sitk_grayscale_ED_high_resampling_resize_ukbb,os.path.join(not_resampling,"UKBB_HIGH_RESAMPLING_SEG_ED.nii.gz"))
                        sitk.WriteImage(sitk_grayscale_ES_high_resampling_high_resize_ukbb,os.path.join(not_resampling,"UKBB_HIGH_RESAMPLING_SEG_ES.nii.gz"))
      
                        cont_patients += 1

                        #if cont_patients> 0: 
                            #break
        #except:
        #    shutil.rmtree(os.path.join(new_path,patient)) 
        #    list_not.append(patient)
        #    continue
    tot_num =  int(len(ukbb_patients)) - int(len(list_not))
    return tot_num


if __name__ == "__main__":

    size_high   = 90 # only for resampling
    size_low    = 90 # only for resampling
    #############################################################################
    ##############
    # GSCAN DATA #
    ##############
    low_files_syn                  = ["lvsa_SR_ED.nii.gz","lvsa_SR_ES.nii.gz"]
    low_files_low                  = ["LVSA_ED.nii.gz","LVSA_ES.nii.gz"]
    ##########################################################################
    high_files                     = ["lvsa_ED.gipl","lvsa_ES.gipl"]
    seg_high1                      = ["seg_lvsa_ED_high_res.nii.gz","seg_lvsa_ES_high_res.nii.gz"]
    seg_high2                      = ["seg_lvsa_SR_ED.nii.gz","seg_lvsa_SR_ES.nii.gz"]
    low_path                       = "./"
    high_path                      = "./"
    #############
    # UKBB DATA #
    #############
    ukbb_path                      = "./"
    save_path                      = "./" 
    low_files_ukbb_grayscale       = ["lvsa_SR_ED.nii.gz","lvsa_SR_ES.nii.gz"]
    ##########################################################################
    low_files_ukbb_grayscale_high  = ["lvsa_SR_ED.nii.gz","lvsa_SR_ES.nii.gz"]
    low_files_ukbb_grayscale_low   = ["lvsa_ED.nii.gz","lvsa_ES.nii.gz"]
    ##########################################################################
    seg_files_ukbb_high            = ["seg_lvsa_SR_ED.nii.gz","seg_lvsa_SR_ES.nii.gz"]
    seg_file_ukbb_low              = ["LVSA_seg_ED.nii.gz","LVSA_seg_ES.nii.gz"] 
    mean_std_dir                   = "./"    
    low_files_seg                  = ["LVSA_seg_ED.nii.gz","LVSA_seg_ES.nii.gz"]        
    ####################
    # Get mean and std #
    ####################
    #mkdir_dir(mean_std_dir)
    #MeanStd (low_path,high_path,ukbb_path,mean_std_dir,low_files,high_files,seg_high1,\
    #        seg_high2,size_high,low_files_ukbb_grayscale_high,low_files_ukbb_grayscale_low,\
    #        seg_files_ukbb_high,seg_file_ukbb_low,size_low)
    #############################################################################

    tot_num = getnii (low_path,high_path,ukbb_path,save_path,low_files_syn,low_files_low,high_files,seg_high1,\
                       seg_high2,size_high,low_files_ukbb_grayscale_high,low_files_ukbb_grayscale_low,\
                       seg_files_ukbb_high,seg_file_ukbb_low,size_low,mean_std_dir,low_files_seg)
    
    print("\n ... Number of total patients is: " + str(tot_num))

    #get_low_res_genscan()
    #############################################################################
