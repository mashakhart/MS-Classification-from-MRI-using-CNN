import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

datapath = r'C:\Users\mkara\OneDrive\Desktop\MS and conditions'
classes_order = ['Alzheimers', 'Healthy', 'MS', 'Parkinsons', 'TBI'] #classes in order of how they appear in direcotry
dataset = ImageFolder(datapath,transform = transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor(),
transforms.Grayscale(num_output_channels=1) ])) #resizes images, converts to tensor, and makes grayscale 

print(dataset.class_to_idx) # prints out: {'Alzheimers': 0, 'Healthy': 1, 'MS': 2, 'Parkinsons': 3, 'TBI': 4}

#ImageFolder is very badly documented online, so this script was just to determine
# what order the subfolders of the directory are assigned to class labels. 
#As I thought, they are in the same order that they appear in the root directory!
# This was also to better understand the output of sklearn's "confusion matrix"