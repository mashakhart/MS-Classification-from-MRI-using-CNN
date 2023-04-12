import os
import random 
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io 

def random_rotation(image_array):
    random_degree = random.uniform(-25,25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array):
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array):
    return image_array[:,::-1]

available_transformations = {'rotate'} 