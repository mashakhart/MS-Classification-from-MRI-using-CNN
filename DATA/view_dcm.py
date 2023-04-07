import pydicom as dicom
import matplotlib.pylab as plt

# specify your image path
image_path = 'TBI\ADNIDOD\0003107\3_Plane_Localizer\2013-05-07_14_21_21.0\I387195\ADNIDOD_0003107_MR_3_Plane_Localizer__br_raw_20130927141417623_8_S199087_I387195.dcm'
ds = dicom.dcmread(image_path)

plt.imshow(ds.pixel_array)