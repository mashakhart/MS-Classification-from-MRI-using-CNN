#forgot to calculate recall and precision within train_and_test, so calculating them from confusion matrix here, 
#instead of rerunning.
import numpy as np

def calc_recall(cm):
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    recall = np.mean(recall)
    return recall

def calc_specificity(cm):
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    specificity = TN/(TN+FP)    
    specificity = np.mean(specificity)
    return specificity

def calc_precision(cm): #just to make sure my calcs for the other two are right
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    precision = np.mean(precision)
    return precision


confusion_matrix = np.array([[ 323,    4,    1,    5,    3],
                             [   3,  666,  118,  134,   25],
                             [   5,  180,  877,   70,    6],
                             [   4,   16,   16, 2171,  109],
                             [   3,   38,   18,  151,  502]])

#this is just for multiclass, since the other ones were easier to calculate, but could use 2-class if needed
recall = calc_recall(confusion_matrix)
specificity = calc_specificity(confusion_matrix)
precision = calc_precision(confusion_matrix)
print('recall is: '+ str(recall) + ' and specificity is: '+str(specificity))
print('precision is: ' + str(precision))
