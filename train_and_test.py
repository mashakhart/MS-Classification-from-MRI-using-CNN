import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from wang_model_CNN import Wang_CNN
from zhang_model_CNN import Zhang_CNN
from simpler_CNN import Simple_CNN
from eight_layer_CNN import Medium_CNN
import matplotlib.pyplot as plt


#read images from dataset using ImageFolder, then split into test and train sets.
def prepare_data(datapath, batch_size, percent_train):
    dataset = ImageFolder(datapath,transform = transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1) ])) #resizes images, converts to tensor, and makes grayscale 

    train_size = int(percent_train*len(dataset)) #num images that are training
    test_size = len(dataset) - train_size #num images that are testing

    #split into train and test
    train_data, test_data = random_split(dataset, [train_size, test_size]) 
    return train_data, test_data

#Creates DataLoaders for the train and test data
def get_data_loaders(train_data, test_data):

    #prepare data loader(combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = True)

    return train_loader, test_loader

#returns accuracy, f1 score, average f1, and confusion matrix for the data
def eval_metrics(ground_truth, predictions, num_classes):

    #dictionary containing the accuracy, precision, f1, avg f1, and confusion matrix for the data
    f1_scores = f1_score(ground_truth, predictions, average=None)
    metrics = {"accuracy": accuracy_score(ground_truth, predictions),
        "f1": f1_scores,
        "average f1": np.mean(f1_scores),
        "confusion matrix": confusion_matrix(ground_truth, predictions),
        }
    
    if num_classes > 2: #if multiclass, set average param to 'micro'; default is binary which doesn't work for multiclass
        metrics['precision'] = precision_score(ground_truth, predictions, average='micro')
    else: #if not multiclass
        metrics['precision'] = precision_score(ground_truth, predictions) #automatically defaults to avg = binary
    return metrics

#Trains the model on the training data using train_loader
def train_cnn(model, loader, optimizer, device, num_classes, silent = True):
    model.train()
    ground_truth = []
    predictions = []
    losses = []
    report_interval = 100

    for data, target in loader:
        data = data.to(device, non_blocking = True)
        target = target.to(device, non_blocking = True)
        logits = model(data) #perform forward pass
        loss = F.cross_entropy(logits, target) #calculate loss
        loss.backward() #backward pass, calculate gradient
        optimizer.step() #update model params
        optimizer.zero_grad() #call zero_grad

        losses.append(loss.item())
        ground_truth.extend(target.tolist())
        predictions.extend(logits.argmax(dim=-1).tolist())

        #print metrics
        if not silent and i > 0 and i % report_interval == 0:
            print(
                "\t[%06d/%06d] Loss: %f"
                % (i, len(loader), np.mean(losses[-report_interval:]))
            )

    return np.mean(losses), eval_metrics(ground_truth, predictions, num_classes)

#Tests the model on the testing data using test_loader
def test_cnn(model, loader, device, num_classes):

    model.eval()
    ground_truth = []
    predictions = []
    losses = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking = True)
            target = target.to(device, non_blocking = True)
            logits = model(data) #perform forward pass
            loss = F.cross_entropy(logits, target) #calculate loss;

            losses.append(loss.item())
            ground_truth.extend(target.tolist())
            predictions.extend(logits.argmax(dim=-1).tolist())

    return np.mean(losses), eval_metrics(ground_truth, predictions, num_classes)


#Runs the training and testing loops on the dataset
def train_and_test(hyperparams, model_type, datapath, batch_size, percent_train, num_classes):

    if model_type == "Zhang":
        model = Zhang_CNN(num_classes)
    elif model_type == "Wang":
        model = Wang_CNN(num_classes)
    elif model_type == "Simple": #if simple (this performs better!! because dataset is smaller, and because no data augmentation)
        model = Simple_CNN(num_classes)
    else:# if model_type == "Medium": #slightly more complex, 8-layer model
        model = Medium_CNN(num_classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    #Create the optimizer (maybe add l2 weight decay later?)
    optimizer = optim.SGD(model.parameters(), lr=hyperparams['learning rate'], momentum = hyperparams['momentum'], weight_decay = hyperparams['l2'])

    #Get data:
    train_data, test_data = prepare_data(datapath, batch_size, percent_train)
    loader_train, loader_test = get_data_loaders(train_data, test_data)

    #Prepare metrics
    train_losses = []
    train_acc = []
    train_prec = []
    test_losses = []
    test_acc = []
    test_prec = []

    #Train and validate
    for i in range(hyperparams['epochs']):
        print("Epoch #%d" % i)

        print("Training..")
        loss_train, metrics_train = train_cnn(model, loader_train, optimizer,device, num_classes, silent= True)
        print("Training loss: ", loss_train)
        print("Training metrics:")
        for k, v in metrics_train.items():
            print("\t", k, ": ", v)
        
        print("Testing..")
        loss_test, metrics_test = test_cnn(model, loader_test, device, num_classes)
        print("Testing loss: ", loss_test)
        print("Testing metrics:")
        for k, v in metrics_test.items():
            print("\t", k, ": ", v)

        train_losses.append(loss_train)
        test_losses.append(loss_test)
        train_acc.append(metrics_train['accuracy'])
        test_acc.append(metrics_test['accuracy'])
        train_prec.append(metrics_train['precision'])
        test_prec.append(metrics_test['precision'])


    print("Done!")

    eval_dict = {"train losses": train_losses, "train accuracies": train_acc, "train precisions": train_prec,
                "test losses": test_losses, "test accuracies": test_acc, "test precisions": test_prec}
    return eval_dict

#plotting results; good to visualize
def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, '-bx')
    plt.plot(test_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Testing'])
    plt.title('Train and Test Loss vs. Epochs')
    plt.show()

def plot_accuracies(train_accuracies, test_accuracies):
    plt.plot(train_accuracies, '-bx')
    plt.plot(test_accuracies, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Testing'])
    plt.title('Train and Test Accuracy vs. Epochs')
    plt.show()

def plot_precision(train_precisions, test_precisions):
    plt.plot(train_precisions, '-bx')
    plt.plot(test_precisions, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.legend(['Training', 'Testing'])
    plt.title('Train and Test Precision vs. Epochs')
    plt.show()

#I have different folders based on what script I run 
#(sort_data_MS_conditions.py, sort_data_MS_healthy.py, sort_data_MS_other.py)
def get_datapath(type):
    if type == 'MS vs healthy':
        datapath = r'C:\Users\mkara\OneDrive\Desktop\MS and healthy' #these paths are for my computer. Change to a local path for your computer
        classes = ['healthy', 'MS']
        num_classes = 2
    elif type == 'MS vs other':
        datapath = r'C:\Users\mkara\OneDrive\Desktop\MS and other' #these paths are for my computer. Change to a local path for your computer
        classes = ['MS-negative', 'MS-positive']
        num_classes = 2
    else: # if MS vs other conditions
        datapath = r'C:\Users\mkara\OneDrive\Desktop\MS and conditions' #these paths are for my computer. Change to a local path for your computer
        classes = ['Alzheimers', 'Healthy', 'MS', 'Parkinsons', 'TBI']
        num_classes = 5
    return datapath, classes, num_classes

hyperparams = {"epochs": 15, "learning rate":0.01, "momentum": 0.9, "l2": 1e-6} 
datapath, classes, num_classes = get_datapath('MS vs other') #can change to "MS vs healthy" or "MS vs conditions"
batch_size = 10 
percent_train = 0.80
model_type = "Simple" #change to "Simple" for the Simple_CNN, "Zhang" for the Zhang_CNN, and "Wang" for the Wang_CNN

eval_dict = train_and_test(hyperparams, model_type, datapath, batch_size, percent_train, num_classes)
plot_loss(eval_dict['train losses'], eval_dict['test losses'])
plot_accuracies(eval_dict['train accuracies'], eval_dict['test accuracies'])
plot_precision(eval_dict['train precisions'], eval_dict['test precisions'])

