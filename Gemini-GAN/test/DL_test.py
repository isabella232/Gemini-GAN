##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################

from utils import *

device = torch.device("cuda")


###############################################################################
# Before starting the code please configure the paths at the end of the code  #
###############################################################################


def save_gt(path_input_SR,path_input_LR,fr,typemode,path_save):
    nim_input_SR_open    = nib.load(path_input_SR)
    nim_input_LR_open    = nib.load(path_input_LR)
    path_save_sr_input   = os.path.join(path_save,"SR_"+typemode+"_INPUT_"+fr+".nii.gz")
    path_save_lr_input   = os.path.join(path_save,"LR_"+typemode+"_INPUT_"+fr+".nii.gz")
    nib.save(nim_input_SR_open, path_save_sr_input)
    nib.save(nim_input_LR_open, path_save_lr_input)


def save_predictions(path_input_SR,fr,typemode,path_save,prediction):
    nim_input_SR_open    = nib.load(path_input_SR)
    prediction         = np.transpose(prediction, (2, 1, 0))
    nim_pred_SR_open   = nib.Nifti1Image(prediction, nim_input_SR_open.affine, nim_input_SR_open.header)
    path_save_sr_pred  = os.path.join(path_save,"SR_"+typemode+"_PREDICTION_"+fr+".nii.gz")
    nib.save(nim_pred_SR_open,  path_save_sr_pred)


def overallDice(name_G,pathsave,path_valid,type_data_valid,pathCode,path_model_seg,in_channels,out_channels):
    patients            = open_txt(os.path.join(pathCode,"dataset_txt",type_data_valid+".txt"))
    pre                 = preprocessing(400,out_channels)
    model               = load(name_G,path_model_seg,in_channels,out_channels)
    #######################
    wrong_patient = []
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
 
    for i in tqdm(range(len(patients))):
        ########################################################################################
        save_path = os.path.join(pathsave,patients[i])
        ########################################################################################
        good_path_dir = pathsave
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
        pre_low_gray_input_ED    = Variable(get_torch(pre.pad_seq(low_gray_input_ED))).cuda()   
        pre_low_gray_input_ES    = Variable(get_torch(pre.pad_seq(low_gray_input_ES))).cuda()  
        ########################################################################################
        output_gray_ED,\
        output_seg_ED   = model(pre_low_gray_input_ED)
        output_gray_ES,\
        output_seg_ES   = model(pre_low_gray_input_ES)
        seg_ED          = segPredict(output_seg_ED)[0]
        seg_ES          = segPredict(output_seg_ES)[0]
        img_ED          = output_gray_ED.cpu().data.numpy()[0]
        img_ES          = output_gray_ES.cpu().data.numpy()[0]
        seg_ED          = pre.getRestoreImg(seg_ED,hight_seg_input_ED)
        seg_ES          = pre.getRestoreImg(seg_ES,hight_seg_input_ES)
        img_ED          = pre.getRestoreImg(img_ED,hight_gray_input_ED)
        img_ES          = pre.getRestoreImg(img_ES,hight_gray_input_ES)
        endo_ed,myo_ed,rv_ed =  DiceEval(seg_ED,hight_seg_input_ED)
        endo_es,myo_es,rv_es =  DiceEval(seg_ES,hight_seg_input_ES)
        ######################################
        good_path = os.path.join(good_path_dir,patients[i])
        mkdir_dir(good_path)
        save_gt(path_hight_gray_input_ED,path_low_gray_input_ED,"ED","GRAY",good_path)
        save_gt(path_hight_gray_input_ES,path_low_gray_input_ES,"ES","GRAY",good_path)
        save_gt(path_hight_seg_input_ED,path_low_seg_input_ED,"ED","SEG",good_path)
        save_gt(path_hight_seg_input_ES,path_low_seg_input_ES,"ES","SEG",good_path)
        save_predictions(path_hight_gray_input_ED,"ED","GRAY",good_path,img_ED)
        save_predictions(path_hight_gray_input_ES,"ES","GRAY",good_path,img_ES)
        save_predictions(path_hight_seg_input_ED,"ED","SEG",good_path,seg_ED)
        save_predictions(path_hight_seg_input_ES,"ES","SEG",good_path,seg_ES)
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
    es_std_psnr_global = np.std(removeNaN(es_std_psnr))
    ###
    ed_mean_ssim_global = np.mean(removeNaN(ed_mean_ssim))
    ed_std_ssim_global = np.std(removeNaN(ed_std_ssim))
    ###
    es_mean_ssim_global = np.mean(removeNaN(es_mean_ssim))
    es_std_ssim_global = np.std(removeNaN(es_std_ssim))
    ###########################################
    str_save = "... Total patients " + str(len(ed_mean_psnr))                                   + "\n" + \
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
    text_file = open(os.path.join(pathsave,name_G + "_table.txt"), "w")
    text_file.write(str_save)
    text_file.close()


def overallDiceOnlySeg(name_G,pathsave,path_valid,type_data_valid,pathCode,path_model_seg,in_channels,out_channels):
    patients            = open_txt(os.path.join(pathCode,"dataset_txt",type_data_valid+".txt"))
    pre                 = preprocessing(400,out_channels)
    model               = load(name_G,path_model_seg,in_channels,out_channels)
    str_save            = ""
    #######################
    wrong_patient = []
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
    for i in tqdm(range(len(patients))):
        ########################################################################################
        save_path = os.path.join(pathsave,patients[i])
        ########################################################################################
        good_path_dir = pathsave
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
        pre_low_gray_input_ED    = Variable(get_torch(pre.pad_seq(low_gray_input_ED))).cuda()   
        pre_low_gray_input_ES    = Variable(get_torch(pre.pad_seq(low_gray_input_ES))).cuda()  
        ########################################################################################
        output_seg_ED   = model(pre_low_gray_input_ED)
        output_seg_ES   = model(pre_low_gray_input_ES)
        seg_ED          = segPredict(output_seg_ED)[0]
        seg_ES          = segPredict(output_seg_ES)[0]
        seg_ED          = pre.getRestoreImg(seg_ED,hight_seg_input_ED)
        seg_ES          = pre.getRestoreImg(seg_ES,hight_seg_input_ES)
        endo_ed,myo_ed,rv_ed =  DiceEval(seg_ED,hight_seg_input_ED)
        endo_es,myo_es,rv_es =  DiceEval(seg_ES,hight_seg_input_ES)
        ######################################
        good_path = os.path.join(good_path_dir,patients[i])
        mkdir_dir(good_path)
        save_gt(path_hight_gray_input_ED,path_low_gray_input_ED,"ED","GRAY",good_path)
        save_gt(path_hight_gray_input_ES,path_low_gray_input_ES,"ES","GRAY",good_path)
        save_gt(path_hight_seg_input_ED,path_low_seg_input_ED,"ED","SEG",good_path)
        save_gt(path_hight_seg_input_ES,path_low_seg_input_ES,"ES","SEG",good_path)
        save_predictions(path_hight_seg_input_ED,"ED","SEG",good_path,seg_ED)
        save_predictions(path_hight_seg_input_ES,"ES","SEG",good_path,seg_ES)
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
    str_save = "... Total patients " + str(len(ed_global_endo_mean))                            + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ED  "      + str(global_mean_endo_ed) + "/" + str(global_std_endo_ed) + "\n" + \
               " ... MYO ED  "       + str(global_mean_myo_ed)  + "/" + str(global_std_myo_ed)  + "\n" + \
               " ... RV ED  "        + str(global_mean_rv_ed)   + "/" + str(global_std_rv_ed)   + "\n" + \
               " -------------------------------------------------------------------------  "   + "\n" + \
               " ... ENDO ES  "      + str(global_mean_endo_es) + "/" + str(global_std_endo_es) + "\n" + \
               " ... MYO ES  "       + str(global_mean_myo_es)  + "/" + str(global_std_myo_es)  + "\n" + \
               " ... RV ES  "        + str(global_mean_rv_es)   + "/" + str(global_std_rv_es)

    text_file = open(os.path.join(pathsave,name_G + "_table.txt"), "w")
    text_file.write(str_save)
    text_file.close()


def overallDiceOnlyImg(name_G,pathsave,path_valid,type_data_valid,pathCode,path_model_seg,in_channels,out_channels):
    patients            = open_txt(os.path.join(pathCode,"dataset_txt",type_data_valid+".txt"))
    pre                 = preprocessing(400,out_channels)
    model               = load(name_G,path_model_seg,in_channels,out_channels)
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
 
    for i in tqdm(range(len(patients))):
        ########################################################################################
        save_path = os.path.join(pathsave,patients[i])
        ########################################################################################
        good_path_dir = pathsave
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
        pre_low_gray_input_ED    = Variable(get_torch(pre.pad_seq(low_gray_input_ED))).cuda()   
        pre_low_gray_input_ES    = Variable(get_torch(pre.pad_seq(low_gray_input_ES))).cuda()  
        ########################################################################################
        output_gray_ED   = model(pre_low_gray_input_ED)
        output_gray_ES   = model(pre_low_gray_input_ES)
        img_ED          = output_gray_ED.cpu().data.numpy()[0]
        img_ES          = output_gray_ES.cpu().data.numpy()[0]
        img_ED          = pre.getRestoreImg(img_ED,hight_gray_input_ED)
        img_ES          = pre.getRestoreImg(img_ES,hight_gray_input_ES)
        ######################################
        good_path = os.path.join(good_path_dir,patients[i])
        mkdir_dir(good_path)
        save_gt(path_hight_gray_input_ED,path_low_gray_input_ED,"ED","GRAY",good_path)
        save_gt(path_hight_gray_input_ES,path_low_gray_input_ES,"ES","GRAY",good_path)
        save_predictions(path_hight_gray_input_ED,"ED","GRAY",good_path,img_ED)
        save_predictions(path_hight_gray_input_ES,"ES","GRAY",good_path,img_ES)
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
    ###########################################
    ed_mean_psnr_global = np.mean(removeNaN(ed_mean_psnr))
    ed_std_psnr_global  = np.std(removeNaN(ed_std_psnr))
    ###################################################
    es_mean_psnr_global = np.mean(removeNaN(es_mean_psnr))
    es_std_psnr_global = np.std(removeNaN(es_std_psnr))
    ###################################################
    ed_mean_ssim_global = np.mean(removeNaN(ed_mean_ssim))
    ed_std_ssim_global = np.std(removeNaN(ed_std_ssim))
    ###################################################
    es_mean_ssim_global = np.mean(removeNaN(es_mean_ssim))
    es_std_ssim_global = np.std(removeNaN(es_std_ssim))
    ###########################################
    str_save = "... Total patients " + str(len(ed_mean_psnr))                               + "\n" + \
               " ... PSNR ED  " + str(ed_mean_psnr_global) + "/" + str(ed_std_psnr_global)  + "\n" + \
               " ... SSIM ED  " + str(ed_mean_ssim_global) + "/" + str(ed_std_ssim_global)  + "\n" + \
               " ... PSNR ES  " + str(es_mean_psnr_global) + "/" + str(es_std_psnr_global)  + "\n" + \
               " ... SSIM ES  " + str(es_mean_ssim_global) + "/" + str(es_std_ssim_global)
    text_file = open(os.path.join(pathsave,name_G + "_table.txt"), "w")
    text_file.write(str_save)
    text_file.close()


if __name__ == "__main__":


    path_code       = "" # Folder where is the list of patients (named dataset_txt)
    path_valid      = "" # Folder where is LR/HR dataset 
    pathsave        = "" # Folder where to save  
    models_path     = "" # folder where the models are saved
    #########################
    
    #########################
    model_type = "img_SR" # models type
    #########################
    
    #########################
    type_data  = "test" # type of datase
    path_model = os.path.join(models_path,model_type,"generator.pt")
    save_folder_path = os.path.join(pathsave,model_type)
    mkdir_dir(save_folder_path)
    #########################
        
    if model_type   == "unet_seg":
        overallDiceOnlySeg(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

    elif model_type == "unet_img":
        overallDiceOnlyImg(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

    elif model_type == "joint_unet":
        overallDice(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

    elif model_type == "joint_unet_gan":
        overallDice(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

    elif model_type == "joint_SR_gan":
        overallDice(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

    elif model_type == "img_SR":
        overallDiceOnlyImg(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

    elif model_type == "seg_SR":
        overallDiceOnlySeg(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

    elif model_type == "img_SR_gan":
        overallDiceOnlyImg(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

    elif model_type == "seg_SR_gan":
        overallDiceOnlySeg(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

    elif model_type == "joint_double_SR_gan":
        overallDice(model_type,save_folder_path,path_valid,type_data,path_code,path_model,100,100)

