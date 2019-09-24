#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from itertools import product


# In[10]:


print(f"{torch.__version__}")


# In[11]:


#to check if gpu is available
trainOnGPU = torch.cuda.is_available()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if not trainOnGPU:
    print("CUDA is not available.Training on CPU")

else:
    print("CUDA is available")


# In[12]:


#number of subprocesses to use for data loading
numOfWorkers = 2

#convert the data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


#get the training and test datasets
init_train_data = datasets.CIFAR10('data',train = True,download = True,transform = transform)
testData = datasets.CIFAR10('data',train = False,download = True,transform = transform)


#get training indices to be used for validation
numOfTrainingSamples = len(init_train_data)
validationSize = 0.2
trainingSize = 1 - validationSize
splits = [int(trainingSize*numOfTrainingSamples), int(validationSize*numOfTrainingSamples)]
# splits = splits.astype(int)
print(f"{splits}")
trainData, valData = dataset.random_split(init_train_data, splits)



#specify the image classes
classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


# In[13]:


#defining the architecture

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
    


parameters = dict(
	lr = [0.01,0.001,0.02,0.002],
	batch_size = [10,100,1000],
	shuffle = [True, False],
)

param_values = [v for v in parameters.values()]

nEpochs = 10
for lr, batch_size, shuffle in product(*param_values):
	print(lr, batch_size, shuffle)
	comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'

	trainLoader = torch.utils.data.DataLoader(trainData,batch_size = batch_size, shuffle=shuffle, num_workers=numOfWorkers)
	validLoader = torch.utils.data.DataLoader(trainData,batch_size= batch_size, shuffle=shuffle, num_workers=numOfWorkers)
	testLoader = torch.utils.data.DataLoader(testData,batch_size=batch_size, num_workers=numOfWorkers)

	comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'
	tb = SummaryWriter(comment=comment)

	validLossMin = np.Inf #initial loss value, it will be updated as we train the model

	model = MyConvNet()
	print(model)
	# tempImg, tempTarget = next(iter(trainLoader))
	#move tensors to GPU if CUDA is available
	# model = nn.DataParallel(model)
	model.to(device)
	# tb.add_graph(model,tempImg.to(device))
	#specify the loss function
	lossCriterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(),lr=lr)#,lr_decay=0.001)
	#specify the Optimizer
	print(optimizer)
	# ### Train the Network

	for epoch in range(1,nEpochs+1):
	    
	    #variables to keep track of training and validation loss
	    trainLoss = 0.0
	    validLoss = 0.0
	    
	    model.train() #switch the model to training mode
	    
	    print("epoch :",epoch," of ",nEpochs)
	    
	    for data,target in trainLoader:
	        
	        #move the tensors to GPU if CUDA is available
	        data,target = data.to(device),target.to(device)
	            
	        #clear the gradients of all optimized variables
	        optimizer.zero_grad()
	        
	        #forward pass : compute the predicted output for present data
	        predictedOutput = model(data)
	        
	        #calculate the batch loss for present data
	        loss = lossCriterion(predictedOutput,target)
	        
	        #backward pass : compute the gradient of the loss with respect to model parameters
	        loss.backward()
	        
	        #perform optimization step for parameters updation
	        optimizer.step()
	        
	        #update training loss
	        trainLoss += loss.item()*batch_size
	        
	        
	    #validation step
	    
	    #switch the model to evaluation mode
	    model.eval()
	    
	    for data,target in validLoader:
	        
	        #move the tensors to GPU if CUDA is available
	        data,target = data.to(device),target.to(device)
	            
	        #perform the forward pass
	        output = model(data)
	        
	        #calculate the batch loss
	        loss= lossCriterion(output,target)
	        
	        #update the average validation loss
	        validLoss += loss.item()*batch_size
	        
	    
	    
	    #calculate average losses
	    trainLoss = trainLoss / len(trainLoader.dataset)
	    validLoss = validLoss / len(validLoader.dataset)
	    
	    
	    print('Epoch: {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(epoch,trainLoss,validLoss))
	    
	    #save the model if validation loss has decreased
	    
	    if validLoss <= validLossMin:
	        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving Model...'.format(
	        validLossMin,validLoss))
	        
	        torch.save(model.state_dict(),'model_cifar.pt')
	        validLossMin = validLoss
	        
	        
	    tb.add_scalar("Train Loss",trainLoss,epoch)
	    tb.add_scalar("Valid Loss",validLoss,epoch)

	    for name, weight in model.named_parameters():
	    	tb.add_histogram(name, weight, epoch)
	    	tb.add_histogram(f'{name}.grad', weight.grad, epoch)

	tb.close()
	        
	        
	# model.load_state_dict(torch.load('model_cifar.pt'))
	# ### test the trained network
	testLoss = 0.0
	classCorrect = list(0.0 for i in range(10))
	classTotal = list(0.0 for i in range(10))

	#switch the model in evaluation mode
	model.eval()
	with torch.no_grad():
	    
	    #iterate over the test data
	    for data,target in testLoader:

	        #move the tensors to GPU if CUDA is available


	        data,target = data.to(device),target.to(device)

	        #forward pass: to compute the predicted labels
	        output = model(data)

	        #calculate the batch loss
	        loss = lossCriterion(output,target)

	        #update test loss
	        testLoss += loss.item()*data.size(0)

	        #convert the output probabilities to predicted class
	        _,pred = torch.max(output,1)

	        #compare predictions to true label
	        correctTensor = pred.eq(target.data.view_as(pred))

	        correct = np.squeeze(correctTensor.numpy()) if not trainOnGPU else np.squeeze(correctTensor.cpu().numpy())

	        #calculate the test accuracy for each object class
	        for i in range(batch_size):
	            label = target.data[i]
	            classCorrect[label] += correct[i].item()
	            classTotal[label] +=1
	        
	#average test loss
	testLoss = testLoss / len(testLoader.dataset)
	print('Test Loss: {:.6f}\n'.format(testLoss))


	for i in range(10):
	    if classTotal[i] > 0:
	        print('Test Accuracy of %5s: %2d%% (%2d/%2d)'%(classes[i],100*classCorrect[i]/classTotal[i],
	                                                     np.sum(classCorrect[i]),np.sum(classTotal[i])))
	        
	    else:
	        print('Test Accuracy of %5s: N/A (no training examples)'%(classes[i]))
	        
	print('\n Test Accuracy (Overall): %2d%% (%2d/%2d)'%(
	        100.*np.sum(classCorrect)/np.sum(classTotal),np.sum(classCorrect),np.sum(classTotal)))





