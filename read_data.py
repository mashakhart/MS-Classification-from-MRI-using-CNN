import os
from os.path import join, getsize
import pydicom as dicom

def read_data_file(datapath):
    img_data_array,class_name = create_dataset_PIL(datapath)
    MS_scans = [] #eHealth are .tif and .bmp, Macin are .png, UMCL .png
    healthy_scans = [] # all are .png
    parkinsons_scans = [] # all are .dcm
    alzheimers_scans = [] # files are HDR, "disc image file", GIF
    #schizophrenia_scans = [] # all are .nii.gz
    TBI_scans = [] # all are .dcm

    non_MS_scans = healthy_scans + parkinsons_scans + alzheimers_scans + schizophenia_scans + TBI_scans

    return MS_scans, non_MS_scans

def create_dataset(Datasets):
    img_data_array=[]
    class_name=[]

    for root, dirs, files in os.walk(Datasets): # root = Datasets, dirs = conditions, files  = scans
        print(root, "consumes", end=" ")
        print(sum(getsize(join(root, name)) for name in files), end=" ")
        print("bytes in", len(files), "non-directory files")

        #if file is in valid image format
        if ".jpg" in file or ".png" in file or ".tif" in file or ".bmp" in file or ".img" in file or ".gif" in file or ".hdr" in file:
            #no need to convert!
            image_path = os.path.join(dirpath, name)
            image= np.array(Image.open(image_path))
            image= np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            class_name.append(dir1)

        if ".dcm" in file:
            #convert
            image_path = os.path.join(dirpath, name)
            ds = dicom.dcmread(image_path)
            image = ds.pixel_array.astype('float32')
            image = (np.maximum(image, 0) / image.max()) * 255.0 #scale it
            image = np.uint8(image) # scale it
            image = Image.fromarray(image)
        
        #not including anything else prevents errors from passing a zipped nifti file or a csv to Pytorch!

    return img_data_array , class_name
    

