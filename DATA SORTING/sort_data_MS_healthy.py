import os
from os.path import join, getsize
import numpy as np
from PIL import Image
#################################################################
#                   DATASET FORMAT INFO                         #
#   eHealth are .tif and .bmp, Macin are .png, UMCL .png        #
#   healthy scans all are .png                                  #
#   parkinson's all are .dcm                                    #
#   alzheimer's: files are HDR, "disc image file", GIF          #
#   TBI scans all are .dcm                                      #
#################################################################

def create_dataset(Datasets, MS_path, healthy_path):
    counter = 0
    MS_counter = 0
    healthy_counter = 0
    for root, dirs, files in os.walk(Datasets, topdown = False): # root = Datasets, files  = scans
        for file in files:

                if ".png" in file or ".tif" in file or ".bmp" in file:
                    counter+=1
                    image_path = os.path.join(root, file)
                    final_image = Image.open(image_path)

                    if "MS" in root:
                        MS_counter+=1
                        final_image.save(MS_path+ '\\'+ "MS_" +str(counter) + '.png', 'PNG')
                    else: #if non-MS
                        healthy_counter+=1
                        final_image.save(healthy_path+ '\\'+ "healthy_" + str(counter)+'.png', 'PNG')
                        

    print("_________DATA VALUES:_________")
    print("Total images: "+str(counter))
    print("Total MS images: "+str(MS_counter))
    print("Total Healthy images: "+ str(healthy_counter))

datapath = r"C:\Users\mkara\OneDrive\Desktop\Datasets"
MS_path = r"C:\Users\mkara\OneDrive\Desktop\MS and healthy\MS"
healthy_path = r"C:\Users\mkara\OneDrive\Desktop\MS and healthy\Healthy"

create_dataset(datapath, MS_path, healthy_path)