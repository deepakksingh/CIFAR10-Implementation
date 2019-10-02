import torch.nn as nn
import torch.nn.functional as F

class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet,self).__init__()
        
        #conv1 layer
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        
        
        #max pool1 layer
        self.pool = nn.MaxPool2d(2,2)
        
        #linear layer (64*4*4 --> 500)
        self.fc1 = nn.Linear(64*4*4,500)
        
        #linear layer (500 --> 10)
        self.fc2 = nn.Linear(500,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,20)

        #linear layer (100 --> 10)
        self.fc5 = nn.Linear(20,10)
        
        #dropout layer
        self.dropout = nn.Dropout(0.25)
        
        #
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        #flatten the image
        x = x.view(-1,64*4*4)
        
        #dropout layer
        x = self.dropout(x)
        
        #first hidden layer, with relu activation
        x = F.relu(self.fc1(x))
        
        #another dropout layer
        x = self.dropout(x)
        
        #another hidden layer without relu activation
        x = F.relu(self.fc2(x))

        #dropout 
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = F.relu(self.fc4(x))
        x = self.dropout(x)

        #last hidden layer
        x = self.fc5(x)
        
        
        return x