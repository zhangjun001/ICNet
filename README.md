# Inverse-Consistent Deep Networks for Unsupervised Deformable Image Registration

Pytorch implementation for Inverse-Consistent Deep Networks for Unsupervised Deformable Image Registration.

This package includes 3D deformable image registration tool for brain images. The code was written by Jun Zhang, Tencent AI Healthcare.


# Applications

3D inverse-consistent deformable image registration

# Prerequisites

Linux python 2.7 NVIDIA GPU + CUDA CuDNN

Install pytorch (0.4.0 or 0.4.1) and dependencies from http://pytorch.org/ 

Install SimpleITK with `pip install SimpleITK` 

Install numpy with `pip install numpy`

# Apply
```
cd Code
```
Apply our Pre-trained Model (Note that the image pairs must be linearly aligned before using our code).
It is better to perform the histogram matching according to any one of our provided images. 
```
python Test.py --fixed ../Dataset/image_A.nii.gz --moving ../Dataset/image_B.nii.gz
```

# Train 
If you want to train a model using your own dataset, please perform the following script.
```
python Train.py --datapath yourdatapath
```
You need to perform the histogram matching for you dataset since we emply the `MSE loss` for measuring similarity.
You may need to adjust the parameters of `--inverse`, `--antifold`, and  `--smooth` for your dataset to get better registration performance. 