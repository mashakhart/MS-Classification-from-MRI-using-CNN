# I ran all of my models without adding labels to confusion matrix, so this file
#is to create a labelled confusion matrix without having to rerun the other code. 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np

def create_confusion_matrix(results, classes, title):
    confusion = ConfusionMatrixDisplay(confusion_matrix= results, display_labels= classes)
    confusion.plot()
    plt.title(title)
    plt.show()
    print('confusion matrix created!')

classes = ["Alzheimer's", "Healthy", "MS", "Parkinson's", "TBI"] #can change based on what you run, but doesn't make sense to create for only two classes
                                                                 
#taken from the train_and_test.py outputs
results =  np.array([[ 322,   15,   10,   14 ,   1],
                     [  14,  663,   67,  118,    1],
                     [   7,  230,  873,   81,    1],
                     [  41,  156,   92, 2028,   12],
                     [  71,  145,   79,  253,  154]])

graph = 'Medium' #can make Simple, Zhang, or Wang
regularization = 'with' #can make 'without' if no l2
title = graph + ' CNN ' + regularization + ' l2 regularization'

create_confusion_matrix(results, classes, title)