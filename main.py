import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict

from RunManager import RunManager
from RunBuilder import RunBuilder

from model import MyConvNet


params = OrderedDict(
		lr = [0.01, 0.001],
		batch_size = [1000, 10000],
		shuffle = [True, False],
		num_workers = [1,2,4]
		)


m = RunManager()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'processing on: {device.type}')
if device.type == 'cuda':
	print(f"Number of GPU(s): {torch.cuda.device_count()}")

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
	])

init_train_data = datasets.CIFAR10('data', train=True, download = True, transform = transform)
testData = datasets.CIFAR10('data',train = False, download = True,transform = transform)


#get training indices to be used for validation
numOfTrainingSamples = len(init_train_data)
validationSize = 0.2
trainingSize = 1 - validationSize
splits = [int(trainingSize*numOfTrainingSamples), int(validationSize*numOfTrainingSamples)]
# splits = splits.astype(int)
print(f"{splits}")
train_set, val_set = dataset.random_split(init_train_data, splits)



for run in RunBuilder.get_runs(params):
	network = MyConvNet()
	loader = DataLoader(train_set, batch_size = run.batch_size, num_workers = run.num_workers, shuffle = run.shuffle)
	optimizer = optim.Adam(network.parameters(), lr = run.lr)

	m.begin_run(run, network, loader)
	for epoch in range(5):
		m.begin_epoch()
		for batch in loader:

			images, labels = batch
			preds = network(images)
			loss = F.cross_entropy(preds, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			m.track_loss(loss)
			m.track_num_correct(preds, labels)
		m.end_epoch()

	m.end_run()

m.save('results')

