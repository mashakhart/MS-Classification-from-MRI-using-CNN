import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class Zhang_CNN(nn.Module):   
    def __init__(self, num_classes):
        super(Zhang_CNN, self).__init__()
        #CHANGE RELU TO PRELU!!
        #BUILD 10-LAYER MODEL AS DESCRIBED IN ZHANG ET AL. ARTICLE ON MS CLASSIFICATION
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding =1) # can be reused
        self.conv_1= nn.Conv2d(1, 16, kernel_size=5, stride=3, padding=2)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=2)
        self.conv_3 = nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=0)
        self.conv_4 = nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1)
        self.conv_5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_7 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        #assuming that num_channels = in_channels, and num_filters = out_channels

        #two classifications: 'MS', 'other'
        self.dropout_1 = nn.Dropout(0.4)
        self.FCL_1 = nn.Linear(512, 500) 
        self.dropout_2 = nn.Dropout(0.5)
        self.FCL_2 = nn.Linear(500, 100)
        self.dropout_3 = nn.Dropout(0.5)
        self.FCL_3 = nn.Linear(100, num_classes)

    # Defining the forward pass    
    def forward(self, x):
    #CONV LAYERS ---------------------------------------------------------------
      #1ST conv layer
      logits = self.conv_1(x)
      logits = F.relu(logits)
      logits = self.pool(logits)
    
      #2nd conv layer
      logits = self.conv_2(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)
      
      #3rd conv layer
      logits = self.conv_3(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)

      #4th conv layer
      logits = self.conv_4(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)

      #5th conv layer
      logits = self.conv_5(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)

      #6th conv layer
      logits = self.conv_6(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)

      #7th conv layer
      logits = self.conv_7(logits)
      logits = F.relu(logits)
      logits = self.pool(logits)

    #DROPOUT + FCL LAYERS ------------------------------------------------------
      #1st dropout + FCL layer
      logits = self.dropout_1(logits)
      logits = logits.view(logits.size(0), -1) #doesn't work if not flattened!!
      logits = self.FCL_1(logits)

      #2nd dropout + FCL layer
      logits = self.dropout_2(logits)
      logits = self.FCL_2(logits)

      #3rd dropout FCL layer
      logits = self.dropout_3(logits)
      logits = self.FCL_3(logits)
    #SOFTMAX--------------------------------------------------------------------
      logits = F.log_softmax(logits, dim=1)
      return logits