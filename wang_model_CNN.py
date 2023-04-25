import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class Wang_CNN(nn.Module):   
    def __init__(self, num_classes):
        super(Wang_CNN, self).__init__()

        #BUILD 10-LAYER MODEL AS DESCRIBED IN ZHANG ET AL. ARTICLE ON MS CLASSIFICATION
        self.BN_1 = nn.BatchNorm2d(2)  
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding ='same') # change to stochastic pool if you can!
        self.conv_1= nn.Conv2d(1, 8, kernel_size=3, stride=2, padding='same') 
        self.conv_2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding='same')
        self.conv_3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same')
        self.conv_4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same')
        self.conv_5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same')
        self.conv_6 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.conv_7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.conv_8 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.conv_9 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        self.conv_10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        #assuming that num_channels = in_channels, and num_filters = out_channels

        #two classifications: 'MS', 'other'
        self.FCL_1 = nn.Linear(1024, 20, bias = True) 
        self.dropout_1 = nn.Dropout(0.5)
        self.FCL_2 = nn.Linear(20, 10, bias = True)
        self.dropout_2 = nn.Dropout(0.5)
        self.FCL_3 = nn.Linear(10,num_classes, bias = True)#or 3 if you want to make "MS", "healthy", "other"

    # Defining the forward pass    
    def forward(self, x):
    #CONV LAYERS -------------------------------------------------------
      #1ST conv layer
      logits = self.conv_1(x)
      logits = self.BN(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)

      #2ND conv layer
      logits = self.conv_2(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)

      #3RD conv layer
      logits = self.conv_3(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)

      #4TH conv layer
      logits = self.conv_4(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)

      #5TH conv layer
      logits = self.conv_5(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)

      #6TH conv layer
      logits = self.conv_6(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)

      #7TH conv layer
      logits = self.conv_7(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)

      #8TH conv layer
      logits = self.conv_8(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)

      #9TH conv layer
      logits = self.conv_9(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)

      #10TH conv layer
      logits = self.conv_10(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)

      #11TH conv layer
      logits = self.conv_11(logits)
      logits = self.BN(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)

      #FCL layers ------------------------------------------------------

      logits = self.FCL_1(logits)
      logits = F.relu(logits)
      logits = self.Dropout_1(logits)
      
      logits = self.FCL_2(logits)
      logits = F.relu(logits)
      logits = self.Dropout_2(logits)

      
      logits = self.FCL_3(logits)

    #SOFTMAX------------------------------------------------------------
      logits = F.log_softmax(logits, dim=1)

      return logits