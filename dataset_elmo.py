from torch.utils.data import Dataset
import numpy as np 
import torch
import os
import json
import pickle
from allennlp.commands.elmo import ElmoEmbedder

def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)	

class SentenceDatasetElmo(Dataset):
	def __init__(self, load=False, test=False):
		super(SentenceDatasetElmo, self).__init__()

		use_cuda = torch.cuda.is_available()
		device = torch.device('cuda:0' if use_cuda else 'cpu')

		elmo = ElmoEmbedder(cuda_device=0) 


		self.premise = []
		self.hypothesis = []
		self.label = []
		self.premise_test = []
		self.hypothesis_test = []
		self.label_test = []
		self.max_len = 0
		self.test = test
		self.batch_size = 32

		lab = {
				"neutral" : 0,
				"contradiction" : 1,
				"entailment" : 2
		}


		if(load):
			traindata = './data/snli_1.0/snli_1.0/snli_1.0_train.jsonl'
			testdata = './data/snli_1.0/snli_1.0/snli_1.0_test.jsonl'

			if(test==False):
				with open(traindata, "r") as f:
					i = 0
					for line in f:
						i = i+1
						jsondata = json.loads(line)

						labl = jsondata["gold_label"]

						if labl == "-":
							continue

						self.label.append(lab[labl])
						
						sentence = jsondata["sentence1"]
						sentence = sentence.split(' ')
						if(len(sentence) > self.max_len):
							self.max_len = len(sentence)
						embedding = elmo.embed_sentence(sentence)
						self.premise.append(np.array(embedding[2]))
						sentence = jsondata["sentence2"]
						sentence = sentence.split(' ')
						if(len(sentence) > self.max_len):
							self.max_len = len(sentence)
						embedding = elmo.embed_sentence(sentence)
						self.hypothesis.append(np.array(embedding[2]))
						if(i%1000==0):
							print(i)
				
				file = open('./pickle/sentences_elmo_train.dat', 'wb+')
				pickle.dump((self.premise, self.hypothesis, self.label), file)
				file.close()
				print("done on training data")

			else:
				with open(testdata, "r") as f:
					i = 0
					for line in f:
						i = i+1
						jsondata = json.loads(line)

						labl = jsondata["gold_label"]

						if labl == "-":
							continue

						self.label_test.append(lab[labl])
						
						sentence = jsondata["sentence1"]
						sentence = sentence.split(' ')
						if(len(sentence) > self.max_len):
							self.max_len = len(sentence)
						embedding = elmo.embed_sentence(sentence)
						self.premise_test.append(np.array(embedding[2]))
						sentence = jsondata["sentence2"]
						sentence = sentence.split(' ')
						if(len(sentence) > self.max_len):
							self.max_len = len(sentence)
						embedding = elmo.embed_sentence(sentence)
						self.hypothesis_test.append(np.array(embedding[2]))
						if(i%1000==0):
							print(i)

				
				file = open('./pickle/sentences_elmo_test.dat', 'wb+')
				pickle.dump((self.premise_test, self.hypothesis_test, self.label_test), file)
				file.close()
				print("done on test data")
		
		else:
			if(self.test):
				file = open('./pickle/sentences_elmo_test.dat', 'rb+')
				pickle.load((self.premise_test, self.hypothesis_test, self.label_test), file)
				file.close()
			else:
				file = open('./pickle/sentences_elmo_train.dat', 'rb+')
				pickle.load((self.premise, self.hypothesis, self.label), file)
				file.close()	

	def __len__(self):
		return len(self.premise)

	def __getitem__(self, idx):	
		premise, hypothesis, label = torch.tensor(self.premise[idx]).type(torch.FloatTensor), torch.tensor(self.hypothesis[idx]).type(torch.FloatTensor), torch.tensor(self.label[idx]).type(torch.LongTensor)
		if(self.test):
			premise, hypothesis, label = torch.tensor(self.premise_test[idx]).type(torch.FloatTensor), torch.tensor(self.hypothesis_test[idx]).type(torch.FloatTensor), torch.tensor(self.label_test[idx]).type(torch.LongTensor)
		premise = pad_tensor(premise, self.max_len, 0)
		hypothesis = pad_tensor(hypothesis, self.max_len, 0)
		return premise, hypothesis, label


if __name__ == '__main__':
	SentenceDatasetElmo(load=True, test=False)
	SentenceDatasetElmo(load=True, test=True)
