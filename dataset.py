import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pandas as pd
import os
import random
import pickle


class FlowSet(Dataset):
	"""CICIDS2017 Raw Flow Dataset"""
	
	def __init__(self, df):
		'''
		Args:
		df(DataFrame): DataFrame containing file paths and corresponding labels
		'''
		#self.flows = df.replace(regex=r'BENIGN', value=0).replace(regex=r'^\w+', value=1)
		self.flows = df

	def __len__(self):
		return len(self.flows) 

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		label = self.flows.loc[idx, 'Label']
		flowPath = self.flows.loc[idx, 'fileName']


		# Data Loading and Converting
		f = open(flowPath, 'rb')
		pklFlow = pickle.load(f)
		srcIP = flowPath.split('/')[-1].split('-')[0].split('.')
		srcIP = np.array([int(value) for value in srcIP])
		dstIP = flowPath.split('/')[-1].split('-')[1].split('.')
		dstIP = np.array([int(value) for value in dstIP])
		count = 0
		flow = np.zeros((1,16,16,16))

		for i, packet in enumerate(pklFlow):
			if i < 16:
				packet = bytes.fromhex(packet)
				packet = np.frombuffer(packet, dtype=np.uint8)

				# Anonymization of IP address
				idx = [i for i in range(len(packet)-4) if np.all(np.equal(packet[i:i+len(srcIP)], srcIP))]
				idx.extend([i for i in range(len(packet)-4) if np.all(np.equal(packet[i:i+len(dstIP)], dstIP))])
				packet = np.delete(packet, np.s_[idx[0]:idx[0]+len(srcIP)+len(dstIP)])
				
				# Normalization
				packet = packet / 255
				if len(packet) > 256:
					packet = packet[:256]
				else:
					packet = np.append(packet, np.zeros(256-len(packet)))
				flow[0, i, :, :] = packet.reshape((16,16))
		f.close()

		sample = {'Flow': flow, 'Label': label, 'FlowPath': flowPath}
		return sample


	



def generate_list(rootPath):
	#Attack label settings: FTP-Patator; SSH-Patator; Bot; Dos Hulk; Web Attack \x96 Brute Force
	labelList = []
	pathList = []
	for f in os.listdir(rootPath):
		if f.endswith('csv'):
			if 'Tuesday' in f:
				labelList.append((f,'FTP-Patator'))
				labelList.append((f,'SSH-Patator'))
				pathList.extend(['../CICIDS2017/data/Tuesday', '../CICIDS2017/data/Tuesday'])
			if 'Wednesday' in f:
				labelList.append((f, 'DoS Hulk'))
				pathList.append('../CICIDS2017/data/Wednesday')
			if 'Thursday' in f and 'Morning' in f:
				labelList.append((f, 'Web Attack \x96 Brute Force'))
				pathList.append('../CICIDS2017/data/Thursday/Morning-WebAttacks')
			if 'Friday' in f:
				if 'Morning' in f:
					labelList.append((f, 'Bot'))
					pathList.append('../CICIDS2017/data/Friday/Morning')
	return labelList, pathList
		
def generate_dataset(metaTestLabels, labelList, csvList, pathList, M, T):
	'''
	M: Size of train set
	T: Size of test set
	'''
	metaTestIdxs = [labelList.index(metaTestLabel) for metaTestLabel in metaTestLabels]
	metaTestCsvs = [csvList[metaTestIdx] for metaTestIdx in metaTestIdxs]
	metaTestPaths = [pathList[metaTestIdx] for metaTestIdx in metaTestIdxs]
	'''
	print(metaTestLabels)
	print(labelList)
	print(csvList)
	print(pathList)
	print(metaTestIdxs)
	'''
	for metaTestLabel in metaTestLabels:
		labelList.remove(metaTestLabel)
	for metaTestCsv in metaTestCsvs:
		csvList.remove(metaTestCsv)
	for metaTestPath in metaTestPaths:
		pathList.remove(metaTestPath)
	'''
	print(labelList)
	print(csvList)
	print(pathList)
	'''
	allDf_0 = pd.DataFrame()
	trainDf = pd.DataFrame()
	testDf = pd.DataFrame()
	for i, label in enumerate(labelList):
		path = pathList[i]
		df = pd.read_csv(csvList[i], low_memory=False)
		if i == len(labelList) - 1:
			df_0 = df[df['Label']=='BENIGN'].sample(n=int((M+T)/len(labelList)) + (M+T)%len(labelList))
		else:
			df_0 = df[df['Label']=='BENIGN'].sample(n=int((M+T)/len(labelList)))
		df_0['fileName'] = df_0['fileName'].apply(lambda x: os.path.join(path, x))
		allDf_0 = pd.concat([allDf_0, df_0], ignore_index=True, sort=False)
		allDf_1 = df[df['Label']==label].sample(n=(M+T))
		allDf_1['fileName'] = allDf_1['fileName'].apply(lambda x: os.path.join(path, x))
		testDf_1 = allDf_1[-T:]
		trainDf_1 = allDf_1[:M]
		trainDf = pd.concat([trainDf, trainDf_1], ignore_index=True, sort=False)
		testDf = pd.concat([testDf, testDf_1], ignore_index=True, sort=False)
	testDf_0 = allDf_0[-T:]
	trainDf_0 = allDf_0[M:]
	trainDf = pd.concat([trainDf, trainDf_0], ignore_index=True, sort=False)
	testDf = pd.concat([testDf, testDf_0], ignore_index=True, sort=False)
	trainDf = trainDf.sample(frac=1).reset_index(drop=True)
	testDf = testDf.sample(frac=1).reset_index(drop=True)
	return {'trainDataset': trainDf, 'metaTestPaths':metaTestPaths, 'metaTestLabels': metaTestLabels, 'metaTestCsvs': metaTestCsvs, 'labelList': labelList, 'testDataset':testDf}
	


def generate_fs_trainset(allDatasetDict, K, B):

	#Generate Sample Set
	labelList = allDatasetDict['labelList']
	df = allDatasetDict['trainDataset']
	i = random.randint(0, len(labelList)-1)
	label = labelList[i]
	Sa_0 = df[df['Label']=='BENIGN'].sample(n=K)
	Sa_1 = df[df['Label']==label].sample(n=K)
	Sa = pd.concat([Sa_0, Sa_1], ignore_index=True, sort=False)
	Sa = Sa.sample(frac=1).reset_index(drop=True)
	sampleSet = FlowSet(Sa)
 
	#Generate Query Set
	Q_0 = df[df['Label']=='BENIGN'].sample(n=B)
	Q_1 = df[df['Label']==label].sample(n=B)
	Q = pd.concat([Q_0, Q_1], ignore_index=True, sort=False)
	Q = Q.sample(frac=1).reset_index(drop=True)
	querySet = FlowSet(Q)
	return {'sampleSet':sampleSet, 'querySet':querySet} 

def generate_fs_trainiter(allDatasetDict, K, B):
	suppBatchSize = 2 * K
	qrBatchSize = 2 * B
	datasetDict = generate_fs_trainset(allDatasetDict, K, B)
	sampleSet = datasetDict['sampleSet']
	querySet = datasetDict['querySet']
	sampleLoader = DataLoader(sampleSet, batch_size=suppBatchSize)
	queryLoader = DataLoader(querySet, batch_size=qrBatchSize)
	trIter = iter(sampleLoader)
	qrIter = iter(queryLoader)
	iterDict = {}
	iterDict['trIter'] = trIter
	iterDict['qrIter'] = qrIter
	return iterDict



def generate_fs_regulartestiter(allDatasetDict, label, epoch, K, B):
	df = allDatasetDict['testDataset']
	Sa_1 = df[df['Label']==label][(K+B) * epoch : K * (epoch+1) + B * epoch]
	Sa_0 = df[df['Label']=='BENIGN'][(K+B) * epoch : K * (epoch+1) + B * epoch]
	Sa = pd.concat([Sa_1, Sa_0], ignore_index=True)
	Sa = Sa.sample(frac=1).reset_index(drop=True)
	supportSet = FlowSet(Sa)
 
	Q_1 = df[df['Label']==label][K * (epoch+1) + B * epoch : (K+B) * (epoch+1)]
	Q_0 = df[df['Label']=='BENIGN'][K * (epoch+1) + B * epoch : (K+B) * (epoch + 1)]
	Q = pd.concat([Q_1, Q_0], ignore_index=True)
	Q = Q.sample(frac=1).reset_index(drop=True)
	querySet = FlowSet(Q)

	supportLoader = DataLoader(supportSet, batch_size=2 * K)
	queryLoader = DataLoader(querySet, batch_size=2 * B)
	suppIter = iter(supportLoader)
	qrIter = iter(queryLoader)
	iterDict = {}
	iterDict['suppIter'] = suppIter
	iterDict['qrIter'] = qrIter
	return iterDict



def generate_fs_metatestiter(dfMetaTest, label, epoch, K, B):

	Sa_1 = dfMetaTest[dfMetaTest['Label']==label][(K+B) * epoch : K * (epoch+1) + B * epoch]
	Sa_0 = dfMetaTest[dfMetaTest['Label']=='BENIGN'][(K+B) * epoch : K * (epoch+1) + B * epoch]
	Sa = pd.concat([Sa_1, Sa_0], ignore_index=True)
	Sa = Sa.sample(frac=1).reset_index(drop=True)
	supportSet = FlowSet(Sa)
 
	Q_1 = dfMetaTest[dfMetaTest['Label']==label][K * (epoch+1) + B * epoch : (K+B) * (epoch+1)]
	Q_0 = dfMetaTest[dfMetaTest['Label']=='BENIGN'][K * (epoch+1) + B * epoch : (K+B) * (epoch + 1)]
	Q = pd.concat([Q_1, Q_0], ignore_index=True)
	Q = Q.sample(frac=1).reset_index(drop=True)
	querySet = FlowSet(Q)

	supportLoader = DataLoader(supportSet, batch_size=2 * K)
	queryLoader = DataLoader(querySet, batch_size=2 * B)
	suppIter = iter(supportLoader)
	qrIter = iter(queryLoader)
	iterDict = {}
	iterDict['suppTestIter'] = suppIter
	iterDict['queryTestIter'] = qrIter
	return iterDict



if __name__ == '__main__':
	pd.set_option('display.max_columns', None)
	pd.set_option('max_colwidth',100)
	allDataset = generate_fs_all_dataset('../CICIDS2017/new_data', 20)
	resultSet = generate_fs_dataset(allDataset, 5, 5)
	sampleSet = resultSet['sampleSet']
	sampleLoader = DataLoader(sampleSet, batch_size=25)
	for batch in sampleLoader:
		print(type(batch['Flow']))
		print(batch['Flow'].shape)
