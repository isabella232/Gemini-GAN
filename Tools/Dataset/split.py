

##################################################
# Nicolo Savioli, Ph.D, Imperial Collage London  # 
##################################################

from   sklearn.model_selection import train_test_split
import numpy    as np 
import os 


def write_txt(savelist,savepath):
    with open(savepath, "a") as f:
        for d in range(len(savelist)):
            print(str(savelist[d]))
            f.write(str(savelist[d]) +"\n")

def  open_txt(path):
    data = []
    with open(path, 'r') as f:
        data = [line.strip() for line in f]
    return data


def SplitDatset(datasetFolder,save_fold):
    patientList     = os.listdir(datasetFolder)
    train, tmp_test = train_test_split(patientList, test_size=0.4, random_state=42)
    test,  valid    = train_test_split(tmp_test,    test_size=0.5, random_state=42)
    # train save 
    write_txt(train,os.path.join(save_fold,"train.txt"))
    # valid save
    write_txt(valid,os.path.join(save_fold,"valid.txt"))
    # test save
    write_txt(test,os.path.join (save_fold,"test.txt" ))



if __name__ == "__main__":
    datasetFolder = ""
    save_fold = ""
    SplitDatset(datasetFolder,save_fold)
