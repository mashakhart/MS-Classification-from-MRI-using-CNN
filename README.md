Code for Independent Work submission for the Princeton Computer Science Department, Spring 2023, by Maria Khartchenko, class of 2024.

MS Classification from MRI scans using CNN models

This is a replication of two models: Wang et al. and Zhang et al., to see if they perform well on a more complex dataset consisting of not only MS scans and healthy control scans, but also scans from patients with Alzheimer's, TBI, and Parkinsons.
I also created two of my own models: Simple_CNN (6-layers: 4 conv, 2 FCL) and Medium_CNN (8-layers: 6 conv, 2 FCL) that performed better on a more limited dataset. 

Data_sorting files allow for conversion between .dcm, .gif, .bmp, and .tif into .png, and then sorts them into folders needed for the models. Change the datapaths to fit your computer. 

Please feel free to email me with any questions at <mk38@princeton.edu>.