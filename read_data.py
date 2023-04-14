import os
from os.path import join, getsize
import pydicom as dicom
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

def read_data_file(datapath, img_size):
    img_data_array,class_name = create_dataset(datapath, img_size)
    num_images = len(class_name)
    
    # #save to corresponding folder in cluster
    # for i in range(num_images):
    #     img = img_data_array[i]
    #     if class_name[i] is "MS":
    #         #save to pathway 
    #         img.save('/scratch/network/mk38/Processed Datasets/MS-positive', 'PNG')
    #     else: #if non-MS
    #         #save to other pathway
    #         img.save('/scratch/network/mk38/Processed Datasets/MS-negative', 'PNG')

    # print('Image processing done!')

def create_dataset(Datasets, img_size):
    img_data_array=[]
    class_name=[]

    for root, dirs, files in os.walk(Datasets, topdown = False): # root = Datasets, dirs = conditions, files  = scans
        print(dirs)
        #print(files)
        for directory in dirs:
            for file in files:
#         print(root, "consumes", end=" ")
#         print(sum(getsize(join(root, name)) for name in files), end=" ")
#         print("bytes in", len(files), "non-directory files")

        #if file is in valid image format
                if ".jpg" in file or ".png" in file or ".tif" in file or ".bmp" in file:# or ".img" in file or ".gif" in file: #or ".hdr" in file:
                    #no need to convert!
                    print('a')
                    print(file)
                    image_path = os.path.join(root, file)
                    image= np.array(Image.open(image_path))
                    image = image.astype('float32')
                    image= np.resize(image,(img_size, img_size, 3))
                    image /= 255  
                    img_data_array.append(image)
                    class_name.append(dirs)
                    print('image appended for file_name: ')
                    

                elif ".dcm" in file:
                    #convert
                    print('b')
                    print(file)
                    image_path = os.path.join(root, file)
                    ds = dicom.dcmread(image_path)
                    image = ds.pixel_array.astype('float32')
                    image= np.resize(image,(img_size, img_size, 3))
                    image /= 255  
                    img_data_array.append(image)
                    class_name.append(dirs)
                    print('image appended for file_name: ')
                    #print(files)
                    #print(dirs)
                else:
                    print('c')
                    #print(file)
        #         #not including anything else prevents errors from passing a zipped nifti file or a csv to Pytorch!

    return img_data_array , class_name
    
# datapath = '/scratch/network/mk38/Datasets'

datapath = r"C:\Users\mkara\OneDrive\Desktop\Datasets"
img_size = 128
read_data_file(datapath, img_size)