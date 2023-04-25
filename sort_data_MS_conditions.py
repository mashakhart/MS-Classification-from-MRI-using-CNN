import os
from os.path import join, getsize
import pydicom as dicom
import numpy as np
from PIL import Image
import imageio.v2 as imageio
#################################################################
#                   DATASET FORMAT INFO                         #
#   eHealth are .tif and .bmp, Macin are .png, UMCL .png        #
#   healthy scans all are .png                                  #
#   parkinson's all are .dcm                                    #
#   alzheimer's: files are HDR, "disc image file", GIF          #
#   TBI scans all are .dcm                                      #
#################################################################


def create_dataset(Datasets, MS_path, healthy_path, parkinsons_path, alzheimers_path, TBI_path):
    counter = 0
    MS_counter = 0
    healthy_counter = 0
    parkinsons_counter = 0
    alzheimers_counter = 0
    TBI_counter = 0
    for root, dirs, files in os.walk(Datasets, topdown = False): # root = Datasets, files  = scans
        for file in files:

            if ".png" in file or ".tif" in file or ".bmp" in file or ".gif" in file or ".dcm" in file:
                counter+=1
                image_path = os.path.join(root, file)

                if ".png" in file or ".tif" in file or ".bmp" in file or ".gif" in file:
                    final_image = Image.open(image_path)

                #DCM-to-PNG conversion implemented from https://pycad.co/how-to-convert-a-dicom-image-into-jpg-or-png/
                else: #if ".dcm" in file:
                    ds = dicom.dcmread(image_path)
                    new_image = ds.pixel_array.astype(float)
                    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
                    scaled_image = np.uint8(scaled_image)
                    final_image = Image.fromarray(scaled_image)

                if "MS" in root:
                    MS_counter+=1
                    final_image.save(MS_path+ '\\'+ "MS_" +str(counter) + '.png', 'PNG')
                elif "Healthy" in root: #if non-MS
                    healthy_counter+=1
                    final_image.save(healthy_path+ '\\'+ "healthy_" + str(counter)+'.png', 'PNG')
                elif "Parkinsons" in root: #if non-MS
                    parkinsons_counter+=1
                    final_image.save(parkinsons_path+ '\\'+ "parkinsons_" + str(counter)+'.png', 'PNG')
                elif "Alzheimers" in root: #if non-MS
                    alzheimers_counter+=1
                    final_image.save(alzheimers_path+ '\\'+ "alzheimers_" + str(counter)+'.png', 'PNG')
                else: #if TBI
                    TBI_counter+=1
                    final_image.save(TBI_path+ '\\'+ "TBI_" + str(counter)+'.png', 'PNG')
                        

    print("_________DATA VALUES:_________")
    print("Total images: "+str(counter))
    print("Total MS images: "+str(MS_counter))
    print("Total Healthy images: "+ str(healthy_counter))
    print("Total Parkinson's images: "+ str(parkinsons_counter))
    print("Total Alzheimer's images: "+ str(alzheimers_counter))
    print("Total TBI images: "+ str(TBI_counter))

datapath = r"C:\Users\mkara\OneDrive\Desktop\Datasets"
MS_path = r"C:\Users\mkara\OneDrive\Desktop\MS and conditions\MS"
healthy_path = r"C:\Users\mkara\OneDrive\Desktop\MS and conditions\Healthy"
parkinsons_path =r"C:\Users\mkara\OneDrive\Desktop\MS and conditions\Parkinsons"
alzheimers_path = r"C:\Users\mkara\OneDrive\Desktop\MS and conditions\Alzheimers"
TBI_path = r"C:\Users\mkara\OneDrive\Desktop\MS and conditions\TBI"

create_dataset(datapath, MS_path, healthy_path, parkinsons_path, alzheimers_path, TBI_path)