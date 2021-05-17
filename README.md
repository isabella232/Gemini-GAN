
# Gemini-GAN

> Joint Semi-supervised 3D Super-Resolution and Segmentation with Mixed Adversarial Gaussian Domain Adaptation

![](img/fig1.png)


 Optimising the analysis of cardiac geometry and motion requires accurate three-dimensional representations, however medical imaging applications are conventionally limited to acquiring contiguous cross-sections with low through-plane resolution and inherent spatial misalignment. Super-resolution is used in computer vision tasks to recover high-resolution images from low-resolution inputs but these are trained on transformations of the acquired images and are not optimised for segmentation tasks. Here we propose a semi-supervised algorithm that jointly estimates super-resolved cine greyscale images and their corresponding three-dimensional segmentation using natively high-resolution ground truth. This uses a multi-task supervised generative adversarial network (Gemini-GAN) for super-resolution and segmentation, and an unsupervised variational adversarial mixture autoencoder (V-AMA) for continuous domain adaptation. We evaluate this on three independent populations demonstrating that our Gemini-GAN shows a competitive Dice metric for high-resolution 3D segmentation of 0.87(0.05) in Endocardium End-Diastole (ED) vs 0.84(0.05), 0.85(0.05) and 0.82(0.05) (mean and std), performed with state-of-art UNet, SR-GAN, and SegSR-GAN. The grayscale super-resolution results, achieved jointly with high-resolution 3D segmentation, are still competitive where the PSNR in ED/ES is 26.93(0.59)/27.12(0.60) vs 23.66(0.37)/23.53(0.38) and 19.90(0.36)/20.10(0.29) respectively achieved with state-of-art super-resolution models SR-GAN, and SegSR-GAN. Our V-AMA model was also compared with three state-of-art generative domain adaptation models CycleGAN, BicycleGAN, and MUNIT showing the best domain adaption performances both in segmentation and super-resolution. This demonstrates how joint cardiac image super-resolution and segmentation, trained on near-isotropic ground-truth data, enables precision phenotyping across different domains.  



# Dependencies

Before getting started with the code, it's important to have a working environment with all dependencies satisfied. We recommend using the Anaconda distribution of Python 3.5. 


```bash

cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh

```

PyTorch must be installed, please make sure that cuDNN is installed correctly (https://developer.nvidia.com/cudnn).

```bash
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

```

Then install the following libraries:

```bash
pip install torchvision
pip install opencv-python
pip install matplotlib
pip install tqdm
pip install SimpleITK
pip install vtk
pip install scikit-image
pip install nibabel
pip install pickle5
```

The Mirtk installation (http://mirtk.github.io/getstarted.html) is only need for testing the classic super-risolution methods (I.e, Linear/NN/BSpline) and for for the alignment block in V-AMA code.

# Code

The code is divided into three main folders: Gemini-GAN, V-AMA, and Tools. Where, in the Gemini-GAN folder, is possible to find the main model proposed and the DL comparative models. 
While in the V-AMA folder, there is the main V-AMA model with the comparison models. Finally, In Tools folder, is possible to find general support code useful for this project.

##  Gemini-GAN Folder

In the Gemini-GAN folder, we can also find three python files: Gemini-GAN.py, seg_SRGAN.py, and UNet.py. Before to run the code please make sure to have set the paths as described internally in each file. Then, please run the traning with the following command:

```
python Gemini-GAN.py
```

In the subdirectory test, there are two main python files: Classic_test.py and DL_test.py. 

* With Classic_test.py you can easily test the classic super-resolution methods such as Linear/NN/BSpline

* With DL_test.py you have also the possibility to test the deep learning method (previously trained). Please remember to set all paths in each code before starting. 

Then, start the test with the following command:

```
cd test
python Classic_test.py
python DL_test.py
```

![](img/LR.gif)

*Low Resolution with 3D Segmentation prediction*

![](img/HR.gif)

*Both Super-Resolution and 3D Segmentation prediction*


##  V-AMA Folder

In the V-AMA you can also find the python file v_ama.py which, in addition to containing the code of the proposed V-AMA model, contains the comparative methods MUNIT, CycleGAN, and BicycleGAN.  
Before to run the code make sure to have set the paths as described in the file. After setting the paths please run the traning with the following command:

```
python v_ama.py
```

In the subdirectory test there is the python test file called test.py. Then, after setting the paths you can start the test with the following command:

```
cd test
python test.py
```

Finally, the subdirectory alignment block is used for the spatial alignment between source and target volume before the domain adaption and you will need Mrtk installed.

![](img/fig2.png)

##  Tools Folder

The tools folder is organized in a subdirectory as follow:

* ./4DCine folder contains the code for creating the 4D Cine for both super-resolution and 3D Segmentation.
* ./dataset folder contains the code used for supporting the dataset creation (I.e, train, valid, test).


## Acknowledgments

The study was supported by Bayer AG; Medical Research Council (MC-A658-5QEB0); National Institute for Health Research (NIHR) Imperial College Biomedical Research Centre; British Heart Foundation (NH/17/1/32725, RG/19/6/34387, RE/18/4/34215); Academy of Medical Sciences (SGL015/1006); Mason Medical Research Trust grant; and the Engineering and Physical Sciences Research Council (EP/P001009/1).

Also, some models are been adapted from these repositries:

* UNet: https://github.com/milesial/Pytorch-UNet
* MUNIT, CycleGAN, and BicycleGAN: https://github.com/eriklindernoren/PyTorch-GAN



## License

Distributed under the GNU GENERAL PUBLIC LICENSE license.
