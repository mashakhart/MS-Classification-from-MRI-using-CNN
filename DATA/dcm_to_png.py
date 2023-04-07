import pydicom as dicom
import cv2   

def dcm_to_png(image_path):
# specify your image path
    ds = dicom.dcmread(image_path)

    pixel_array_numpy = ds.pixel_array

    image_format = '.png' # or '.jpg'
    image_path = image_path.replace('.dcm', image_format)

    cv2.imwrite(image_path, pixel_array_numpy)
    print('converted dcm image to png successfully')