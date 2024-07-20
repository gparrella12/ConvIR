[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revitalizing-convolutional-network-for-image/image-dehazing-on-sots-indoor)](https://paperswithcode.com/sota/image-dehazing-on-sots-indoor)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revitalizing-convolutional-network-for-image/image-dehazing-on-sots-outdoor)](https://paperswithcode.com/sota/image-dehazing-on-sots-outdoor)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revitalizing-convolutional-network-for-image/image-dehazing-on-haze4k)](https://paperswithcode.com/sota/image-dehazing-on-haze4k)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revitalizing-convolutional-network-for-image/image-dehazing-on-i-haze)](https://paperswithcode.com/sota/image-dehazing-on-i-haze)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revitalizing-convolutional-network-for-image/image-dehazing-on-o-haze)](https://paperswithcode.com/sota/image-dehazing-on-o-haze)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revitalizing-convolutional-network-for-image/snow-removal-on-snow100k)](https://paperswithcode.com/sota/snow-removal-on-snow100k)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revitalizing-convolutional-network-for-image/snow-removal-on-srrs)](https://paperswithcode.com/sota/snow-removal-on-srrs)


## Revitalizing Convolutional Network for Image Restoration

The official pytorch implementation of the paper **[Revitalizing Convolutional Network for Image Restoration
 (T-PAMI'24)](https://ieeexplore.ieee.org/abstract/document/10571568)**

#### Yuning Cui, Wenqi Ren, Xiaochun Cao, Alois Knoll

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Training and Evaluation
Please refer to respective directories.
## Results [Download]
|Model|Parameters|FLOPs|
|------|-----|-----|
|*ConvIR-S (small)*|5.53M|42.1G|
|**ConvIR-B (base)**| 8.63M|71.22G|
|<u>ConvIR-L (large)</u>| 14.83M |129.34G|

|Task|Dataset|PSNR|SSIM|
|----|------|-----|----|
|**Image Dehazing**|SOTS-Indoor|*41.53*/**42.72**|*0.996*/**0.997**|
||SOTS-Outdoor|*37.95*/**39.42**|*0.994*/**0.996**|
||Haze4K|*33.36*</font>/**34.15**/<u>34.50</u>|*0.99*/**0.99**/<u>0.99</u>|
||Dense-Haze|*17.45*/**16.86**|*0.648*/**0.621**|
||NH-HAZE|*20.65*/**20.66**|*0.807*/**0.802**|
||O-HAZE|*25.25*/**25.36**|*0.784*/**0.780**|
||I-HAZE|*21.95*/**22.44**|*0.888*/**0.887**|
||SateHaze-1k-Thin/Moderate/Thick|*25.11*/*26.79*/*22.65*|*0.978*/*0.978*/*0.950*|
||NHR|*28.85*/**29.49**|*0.981*/**0.983**|
||GTA5|*31.68*/**31.83**|*0.917*/**0.921**|
|**Image Desnowing**|CSD|*38.43*/**39.10**|*0.99*/**0.99**|
||SRRS|*32.25*/**32.39**|*0.98*/**0.98**|
||Snow100K|*33.79*/**33.92**|*0.95*/**0.96**|
|**Image Deraining**|Test100|<u>31.40</u>|<u>0.919</u>|
||Test2800|<u>33.73</u>|<u>0.937</u>|
|**Defocus Deblurring**|DPDD|*26.06*/**26.16**/<u>26.36</u>|*0.810*/**0.814**/<u>0.820</u>|
|**Motion Deblurring**|GoPro|<u>33.28</u>|<u>0.963</u>|
||RSBlur|<u>34.06</u>|<u>0.868</u>|


## Citation
~~~
@article{cui2024revitalizing,
  title={Revitalizing Convolutional Network for Image Restoration},
  author={Cui, Yuning and Ren, Wenqi and Cao, Xiaochun and Knoll, Alois},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}

@inproceedings{cui2023irnext,
  title={IRNeXt: Rethinking Convolutional Network Design for Image Restoration},
  author={Cui, Yuning and Ren, Wenqi and Yang, Sining and Cao, Xiaochun and Knoll, Alois},
  booktitle={International Conference on Machine Learning},
  pages={6545--6564},
  year={2023},
  organization={PMLR}
}
~~~

## Contact
Should you have any problem, please contact Yuning Cui.
