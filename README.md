# MS Classification from MRI scans using CNN models
## Code for Independent Work submission for the Princeton Computer Science Department, Spring 2023, by Maria Khartchenko, class of 2024

This is a replication of two models: Zhang et al. (1) and Wang et al. (2), to see if they perform well on a more complex dataset consisting of not only MS scans and healthy control scans, but also scans from patients with Alzheimer's, TBI, and Parkinsons.
I also created two of my own models: Simple_CNN (6-layers: 4 conv, 2 FCL) and Medium_CNN (8-layers: 6 conv, 2 FCL) that performed better on a more limited dataset. 

Data_sorting files allow for conversion between .dcm, .gif, .bmp, and .tif into .png, and then sorts them into folders needed for the models. Change the datapaths to fit your computer. 

**_Please feel free to email me with any questions at <mk38@princeton.edu>._**


(1): Zhang YD, Pan C, Sun J and Tang C: Multiple Sclerosis identification by convolutional neural networks with dropout and parametric ReLU. J Comput Sci 28: 818 (2018)

(2): Wang SH, Tang C, Sun J, Yang J, Huang C, Phillips P and Zhang YD: Multiple Sclerosis Identification by 14-Layer Convolutional Neural Network with Batch Normalization, Dropout, and Stochastic Pooling. Front Neurosci 12: 818 (2018)
