import torch.nn as nn
import torch
import torch.nn.functional as F


torch.set_printoptions(profile='full')

def conv_block1(inChannels):
	return nn.Sequential(
		nn.Conv3d(inChannels, 128, 2),
		nn.BatchNorm3d(128),
		nn.ReLU(),
		nn.Dropout(p=0.4))

def conv_block2(inChannels):
	return nn.Sequential(
		nn.Conv3d(inChannels, 128, 2),
		nn.BatchNorm3d(128),
		nn.ReLU())

def conv_block3(inChannels):
	return nn.Sequential(
		nn.Conv3d(inChannels, 128, 2),
		nn.BatchNorm3d(128),
		nn.ReLU(),
		nn.MaxPool3d(2),
		nn.Dropout(p=0.2))

class FNet(nn.Module):

	def __init__(self):
		super(FNet, self).__init__()
		self.encoder = nn.Sequential(
			conv_block1(1),
			conv_block2(128),
			conv_block1(128),
			conv_block2(128))
	
	def forward(self, x):
		x = self.encoder(x)
		#return x.view(x.size(0), -1)
		return x

class CNet(nn.Module):
	def __init__(self):
		super(CNet, self).__init__()
		'''
		self.cnet = nn.Sequential(
			conv_block3(256),
			conv_block3(128))
		'''
		self.layer1 = nn.Sequential(
			nn.Conv3d(256, 128, 2),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.MaxPool3d(2),
			nn.Dropout(p=0.2))
		self.layer2 = nn.Sequential(
			nn.Conv3d(128, 128, 2),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.MaxPool3d(2),
			nn.Dropout(p=0.2))
		self.fc1 = nn.Linear(1024, 64)
		self.fc2 = nn.Linear(64, 1)

	def forward(self, x):
		#x = self.cnet(x)
		#print(next(self.fc1.parameters()).is_cuda)
		#print(next(self.fc2.parameters()).is_cuda)
		x = self.layer1(x)
		x = self.layer2(x)
		#x = x.view(80, -1)
		x = x.view(160, -1)  #For 10 shot
		#print(x.device)
		x = self.fc1(x)
		x = self.fc2(x)
		x = torch.sigmoid(x)
		return x
