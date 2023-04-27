import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class Wang_CNN(nn.Module):   
    def __init__(self, num_classes):
        super(Wang_CNN, self).__init__()

        #BUILD 14-LAYER MODEL AS DESCRIBED IN WANG ET AL. ARTICLE ON MS CLASSIFICATION

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding =1) #same padding = kernel size
        self.conv_1= nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same') 
        self.BN_1 = nn.BatchNorm2d(8)
        self.conv_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding='same')
        self.BN_2 = nn.BatchNorm2d(8)
        self.conv_3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same')
        self.BN_3 = nn.BatchNorm2d(16)
        self.conv_4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same')
        self.BN_4 = nn.BatchNorm2d(16)
        self.conv_5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same')
        self.BN_5 = nn.BatchNorm2d(16)
        self.conv_6 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.BN_6 = nn.BatchNorm2d(32)
        self.conv_7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.BN_7 = nn.BatchNorm2d(32)
        self.conv_8 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.BN_8 = nn.BatchNorm2d(32)
        self.conv_9 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        self.BN_9 = nn.BatchNorm2d(64)
        self.conv_10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.BN_10 = nn.BatchNorm2d(64)
        self.conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.BN_11 = nn.BatchNorm2d(64)
        #assuming that num_channels = in_channels, and num_filters = out_channels

        #two classifications: 'MS', 'other'
        self.FCL_1 = nn.Linear(10, 20, bias = True) 
        self.dropout_1 = nn.Dropout(0.5)
        self.FCL_2 = nn.Linear(20, 10, bias = True)
        self.dropout_2 = nn.Dropout(0.5)
        self.FCL_3 = nn.Linear(10,num_classes, bias = True)#or 3 if you want to make "MS", "healthy", "other"

    # Defining the forward pass    
    def forward(self, input):
    #CONV LAYERS -------------------------------------------------------
      #1ST conv layer
      output = self.conv_1(input)
      output = self.BN_1(output)
      output = F.relu(output)
      output = self.pool(output)

      #2ND conv layer
      output = self.conv_2(output)
      output = self.BN_2(output)
      output = F.relu(output)
      output = self.pool(output)

      #3RD conv layer
      output = self.conv_3(output)
      output = self.BN_3(output)
      output = F.relu(output)

      #4TH conv layer
      output = self.conv_4(output)
      output = self.BN_4(output)
      output = F.relu(output)

      #5TH conv layer
      output = self.conv_5(output)
      output = self.BN_5(output)
      output = F.relu(output)
      output = self.pool(output)

      #6TH conv layer
      output = self.conv_6(output)
      output = self.BN_6(output)
      output = F.relu(output)

      #7TH conv layer
      output = self.conv_7(output)
      output = self.BN_7(output)
      output = F.relu(output)

      #8TH conv layer
      output = self.conv_8(output)
      output = self.BN_8(output)
      output = F.relu(output)

      #9TH conv layer
      output = self.conv_9(output)
      output = self.BN_9(output)
      output = F.relu(output)

      #10TH conv layer
      output = self.conv_10(output)
      output = self.BN_10(output)
      output = F.relu(output)

      #11TH conv layer
      output = self.conv_11(output)
      output = self.BN_11(output)
      output = F.relu(output)
      output = self.pool(output)

      #FCL layers ------------------------------------------------------

      output = self.FCL_1(output)
      output = F.relu(output)
      output = self.dropout_1(output)
      
      output = self.FCL_2(output)
      output = F.relu(output)
      output = self.dropout_2(output)

      
      output = self.FCL_3(output)

    #SOFTMAX------------------------------------------------------------
      output = F.log_softmax(output, dim=1)

      return output