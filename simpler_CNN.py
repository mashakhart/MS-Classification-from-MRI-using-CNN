import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class Simple_CNN(nn.Module):   
    def __init__(self, num_classes):
        super(Simple_CNN, self).__init__()

        #SIMPLER, 6-layer model: 4 conv layers and 2 fully connected layers. works better on smaller dataset!

        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding =1)
        self.conv_1= nn.Conv2d(1, 16, kernel_size=5, stride=3, padding=2)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=2)
        self.conv_3 = nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=0)
        self.conv_4 = nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1)

        self.FCL_1 = nn.Linear(256, 100) 
        self.dropout_1 = nn.Dropout(0.4)
        self.FCL_2 = nn.Linear(100, num_classes)

    # Defining the forward pass    
    def forward(self, input):
    #CONV LAYERS ---------------------------------------------------------------
      #1ST conv layer
      output = self.conv_1(input)
      output = F.relu(output)
      output = self.pool(output)
    
      #2nd conv layer
      output = self.conv_2(output)
      output = F.relu(output)
      output = self.pool(output)
      
      #3rd conv layer
      output = self.conv_3(output)
      output = F.relu(output)
      output = self.pool(output)

      #4th conv layer
      output = self.conv_4(output)
      output = F.relu(output)
      output = self.pool(output)

    #DROPOUT + FCL LAYERS ------------------------------------------------------
      #1st dropout + FCL layer
      output = output.view(output.size(0), -1) #doesn't work if not flattened!!
      output = self.FCL_1(output)

      #2nd dropout + FCL layer
      output = self.dropout_1(output)
      output = self.FCL_2(output)

    #SOFTMAX--------------------------------------------------------------------
      output = F.log_softmax(output, dim=1)
      return output