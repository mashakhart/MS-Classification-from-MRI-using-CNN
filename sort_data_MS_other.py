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

def create_dataset(Datasets, img_size, MS_path, non_MS_path):
    counter = 0
    MS_counter = 0
    non_MS_counter = 0
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
                else: #if non-MS
                    non_MS_counter+=1
                    final_image.save(non_MS_path+ '\\'+ "other_" + str(counter)+'.png', 'PNG')
                        

            #     image = np.resize(image,(img_size, img_size, 3))
            #     #image /= 255.0  
            #     image = Image.fromarray((image * 255).astype(np.uint8))
            #     if "MS" in root:
            #         MS_counter+=1
            #         image.save(MS_path+ '\\'+ str(counter) + '.png', 'PNG')
            #     else: #if non-MS
            #         non_MS_counter+=1
            #         image.save(non_MS_path+ '\\'+ str(counter)+'.png', 'PNG')

    print("_________DATA VALUES:_________")
    print("Total images: "+str(counter))
    print("Total MS images: "+str(MS_counter))
    print("Total non-MS images: "+ str(non_MS_counter))

datapath = r"C:\Users\mkara\OneDrive\Desktop\Datasets"
MS_path = r"C:\Users\mkara\OneDrive\Desktop\MS and other\MS-positive"
non_MS_path = r"C:\Users\mkara\OneDrive\Desktop\MS and other\MS-negative"
img_size = 128

create_dataset(datapath, img_size, MS_path, non_MS_path)