# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import sys

def euclidean_dist(x, y):
	'''
	Compute euclidean distance between two tensors
	'''
	# x: N x D
	# y: M x D
	n = x.size(0)
	m = y.size(0)
	d = x.size(1)
	if d != y.size(1):
		raise Exception

	x = x.unsqueeze(1).expand(n, m, d)
	y = y.unsqueeze(0).expand(n, m, d)

	return torch.pow(x - y, 2).sum(2)

def calculate_acc(x, y):
	x = x.max(dim=1)[1]
	acc = x.eq(y).sum().float() / len(x)
	return acc

def test_result_trace(x, labels):
	x = x.max(dim=1)[1]
	acc = torch.where(x==labels, torch.ones_like(x), torch.zeros_like(x))
	acc = acc.sum().float() / len(x)
	#print('x:', x)
	#print('torch.nonzero(labels, as_tuple=False):', torch.nonzero(labels, as_tuple=False))
	#print('torch.nonzero(x, as_tuple=False):', torch.nonzero(x, as_tuple=False))
	#print('labels:', labels)	

	p1 = torch.nonzero(labels, as_tuple=False)
	p2 = torch.nonzero(x, as_tuple=False)
	truePos = len(set(p1.int().reshape(-1).tolist()).intersection(set(p2.int().reshape(-1).tolist())))
	falseNeg = len(set(p1.int().reshape(-1).tolist()).difference(set(p2.int().reshape(-1).tolist())))
	#print('truePos:', truePos)
	#print('falseNeg:', falseNeg)
	dr = truePos / (truePos + falseNeg)
	result = {'dr':dr, 'acc':acc}
	return result
		

def calculate_prototype_pairs(suppInput, queryInput, suppTarget, queryTarget):

	classes = torch.unique(suppTarget)
	nClasses = len(classes)
	nSupport = suppTarget.eq(classes[0].item()).sum().item()
	nQuery = queryTarget.eq(classes[0].item()).sum().item()
	#print('nSupport:', nSupport)  
	#print('nQuery:', nQuery)
	
	def supp_idxs(c):
		return suppTarget.eq(c).nonzero()[:nSupport].squeeze(1)

	supportIdxs = list(map(supp_idxs, classes))
	#print('supportIdxs:', supportIdxs)
	#print('suppInput.shape:', suppInput.shape)  #(10, 128, 12, 12, 12)
	#print('queryInput.shape:', queryInput.shape)  #(40, 128, 12, 12, 12)
	prototypes = torch.stack([suppInput[idxList].mean(0) for idxList in supportIdxs])
	#print('prototypes.shape:', prototypes.shape)  #(2, 128, 12, 12, 12)
	prototypes = prototypes.unsqueeze(0).repeat(nClasses*nQuery, 1, 1, 1, 1, 1)
	#print('prototypes.shape:', prototypes.shape)  #(40, 2, 128, 12, 12, 12)
	queryInput = queryInput.unsqueeze(0).repeat(nClasses,1, 1, 1, 1, 1)
	#print('queryInput.shape:', queryInput.shape)
	queryInput = torch.transpose(queryInput, 0, 1)
	#print('queryInput.shape:', queryInput.shape)
	pairs = torch.cat((prototypes, queryInput), 2).view(-1, 256, 12, 12, 12)
	return pairs