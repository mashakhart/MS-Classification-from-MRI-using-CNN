
import os
from os.path import join, getsize
import pydicom as dicom

def read_data_file(datapath):
    MS_scans = []
    healthy_scans = []
    parkinsons_scans = []
    alzheimers_scans = []
    #schizophrenia_scans = []
    TBI_scans = []

def create_dataset_PIL(Datasets):
    img_data_array=[]
    class_name=[]

    for root, dirs, files in os.walk(Datasets): # root = Datasets, dirs = conditions, files  = scans
        print(root, "consumes", end=" ")
        print(sum(getsize(join(root, name)) for name in files), end=" ")
        print("bytes in", len(files), "non-directory files")

        if ".dcm" in file:
            #convert
        if ".jpg" in file:
            #convert
        if ".tif" in file:
            #convert
            image = Image.open(os.path.join(root, name))
            
        if ".dcm" in file:
            #convert
            image_path = os.path.join(dirpath, name)
            ds = dicom.dcmread(image_path)
            image = ds.pixel_array.astype('float32')
            image = (np.maximum(image, 0) / image.max()) * 255.0 #scale it
            image = np.uint8(image) # scale it
            image = Image.fromarray(image)
        if ".img" in file:
            #convert
        if ".nii.gz" in file: 
            #unzip, and then walk (make walk separate function)
        if ".png" in file:
            image_path = os.path.join(dirpath, name)

            image= np.array(Image.open(image_path))
            image= np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            class_name.append(dir1)

    return img_data_array , class_name
PIL_img_data, class_name=create_dataset_PIL(img_folder)


    non_MS_scans = healthy_scans + parkinsons_scans + alzheimers_scans + schizophenia_scans + TBI_scans

    return MS_scans, non_MS_scans


