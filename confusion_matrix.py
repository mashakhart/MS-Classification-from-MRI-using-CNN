# I ran all of my models without adding labels to confusion matrix, so this file
#is to create a labelled confusion matrix without having to rerun the other code. 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np

def create_confusion_matrix(results, classes):

    disp = ConfusionMatrixDisplay(confusion_matrix= results, display_labels= classes)
    disp.plot()
    plt.show()


classes = ["Alzheimer's", "Healthy", "MS", "Parkinson's", "TBI"] #can change based on what you run, but doesn't make sense to create for only two classes
                                                                 
#taken from the train_and_test.py outputs
results =  np.array([[1152,13,33,168,106],
            [   6, 2029,  591,  690,  309],
            [  33,  706, 2989,  452,  270],
            [  76,  240,  268, 8402,  446],
            [  88,  206,  305, 1452,  759]])



create_confusion_matrix(results, classes)