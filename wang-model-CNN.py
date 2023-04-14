class Wang_CNN(Module):   
    def __init__(self):
        super(Net, self).__init__()

        #BUILD 10-LAYER MODEL AS DESCRIBED IN ZHANG ET AL. ARTICLE ON MS CLASSIFICATION
        self.ReLU = nn.ReLU(inplace=True) # can be reused
        self.BN_1 = nn.BatchNorm2d(2) #NEED TO FIGURE OUT NUM FEATURES #can be reused # batch_size, num channels=1, H , W #add 
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding ='same') # change to stochastic pool
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
        self.FCL_1 = nn.Linear(1024, 20) #unsure about these nums
        self.dropout_1 = nn.Dropout(0.5)
        self.FCL_2 = nn.Linear(20, 10)
        self.dropout_2 = nn.Dropout(0.5)
        self.FCL_3 = nn.Linear(10,2)#or 3 if you want to make "MS", "healthy", "other"

    # Defining the forward pass    
    def forward(self, x):
    #CONV LAYERS -------------------------------------------------------
      #1ST conv layer
      logits = self.conv_1(x)
      logits = self.BN(logits)
      logits = self.ReLU(logits)
      logits = self.pool(logits)

      #2ND conv layer
      logits = self.conv_2(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)
      logits = self.pool(logits)

      #3RD conv layer
      logits = self.conv_3(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)

      #4TH conv layer
      logits = self.conv_4(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)

      #5TH conv layer
      logits = self.conv_5(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)
      logits = self.pool(logits)

      #6TH conv layer
      logits = self.conv_6(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)

      #7TH conv layer
      logits = self.conv_7(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)

      #8TH conv layer
      logits = self.conv_8(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)

      #9TH conv layer
      logits = self.conv_9(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)

      #10TH conv layer
      logits = self.conv_10(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)

      #11TH conv layer
      logits = self.conv_11(logits)
      logits = self.BN(logits)
      logits = self.ReLU(logits)
      logits = self.pool(logits)

      #FCL layers ------------------------------------------------------

      logits = self.FCL_1(logits)
      logits = self.ReLU(logits)
      logits = self.Dropout_1(logits)
      
      logits = self.FCL_2(logits)
      logits = self.ReLU(logits)
      logits = self.Dropout_2(logits)

      
      logits = self.FCL_3(logits)

    #SOFTMAX------------------------------------------------------------
      logits = F.log_softmax(logits, dim=1)

      return logits