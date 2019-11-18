from torch.utils.data import Dataset
import numpy as np 
import torch
import os
import json
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model = bert_model.cuda()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)	

class SentenceDataset(Dataset):
	def __init__(self, load=False, test=False):
		super(SentenceDataset, self).__init__()

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
					for line in f:
						jsondata = json.loads(line)

						labl = jsondata["gold_label"]

						if labl == "-":
							continue

						self.label.append(lab[labl])
						
						sentence = jsondata["sentence1"]
						self.premise.append(sentence)
						sentence = sentence.split(' ')
						if(len(sentence) > self.max_len):
							self.max_len = len(sentence)
						sentence = jsondata["sentence2"]
						self.hypothesis.append(sentence)
						sentence = sentence.split(' ')
						if(len(sentence) > self.max_len):
							self.max_len = len(sentence)

				# self.premise = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in self.premise]
				# indexed_tokens = [tokenizer.convert_tokens_to_ids(sent) for sent in self.premise]
				# segments_ids = [[1] * len(sent) for sent in self.premise]
				# tokens_tensor = [torch.tensor([it]) for it in indexed_tokens]
				# segments_tensors = [torch.tensor([st]) for st in segments_ids]
				# self.premise = []
				# with torch.no_grad():
				# 	for token, segment in zip(tokens_tensor, segments_tensors):
				# 		encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
				# 		embedding = encoded_layers[11][0]
				# 		print(len(embedding))
				# 		print(len(embedding[0]))
				# 		self.premise.append(np.array(embedding))

				# temp = self.premise
				# self.premise = []
				# i=0
				# for text in temp:
				# 	if(i%1000==0):
				# 		print(text, "PREM")
				# 	i = i+1
				# 	marked_text = "[CLS] " + text + " [SEP]"
				# 	tokenized_text = tokenizer.tokenize(marked_text)
				# 	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
				# 	segments_ids = [1] * len(tokenized_text)
				# 	tokens_tensor = torch.tensor([indexed_tokens])
				# 	segments_tensors = torch.tensor([segments_ids])
				# 	bert_model.eval()
				# 	with torch.no_grad():
				# 		encoded_layers, _ = bert_model(tokens_tensor.to(device), segments_tensors.to(device))
				# 		embedding = encoded_layers[11][0]
				# 		self.premise.append(np.array(embedding.cpu()))
				# 		if(i%1000==0):
				# 			print(np.array(embedding.cpu()).shape)


				# temp = self.hypothesis
				# self.hypothesis = []
				# i=0
				# for text in temp:
				# 	if(i%1000==0):
				# 		print(text, "HYP")
				# 	i = i+1
				# 	marked_text = "[CLS] " + text + " [SEP]"
				# 	tokenized_text = tokenizer.tokenize(marked_text)
				# 	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
				# 	segments_ids = [1] * len(tokenized_text)
				# 	tokens_tensor = torch.tensor([indexed_tokens])
				# 	segments_tensors = torch.tensor([segments_ids])
				# 	bert_model.eval()
				# 	with torch.no_grad():
				# 		encoded_layers, _ = bert_model(tokens_tensor.to(device), segments_tensors.to(device))
				# 		embedding = encoded_layers[11][0]
				# 		self.hypothesis.append(np.array(embedding.cpu()))
				# 		if(i%1000==0):
				# 			print(np.array(embedding.cpu()).shape)

				file = open('./pickle/sentences_train.dat', 'wb+')
				pickle.dump((self.premise, self.hypothesis, self.label), file)
				file.close()
				file = open('./pickle/maxlen_train.dat', 'wb+')
				pickle.dump((self.max_len), file)
				file.close()
				print("done on training data")
			else:
				with open(testdata, "r") as f:
					for line in f:
						jsondata = json.loads(line)

						labl = jsondata["gold_label"]

						if labl == "-":
							continue

						self.label_test.append(lab[labl])
						
						sentence = jsondata["sentence1"]
						self.premise_test.append(sentence)
						sentence = sentence.split(' ')
						if(len(sentence) > self.max_len):
							self.max_len = len(sentence)
						sentence = jsondata["sentence2"]
						self.hypothesis_test.append(sentence)
						sentence = sentence.split(' ')
						if(len(sentence) > self.max_len):
							self.max_len = len(sentence)

				# temp = self.premise_test
				# self.premise_test = []
				# i = 0
				# for text in temp:
				# 	if(i%1000==0):
				# 		print(text, "PREM")
				# 	i = i+1
				# 	marked_text = "[CLS] " + text + " [SEP]"
				# 	tokenized_text = tokenizer.tokenize(marked_text)
				# 	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
				# 	segments_ids = [1] * len(tokenized_text)
				# 	tokens_tensor = torch.tensor([indexed_tokens])
				# 	segments_tensors = torch.tensor([segments_ids])
				# 	bert_model.eval()
				# 	with torch.no_grad():
				# 		encoded_layers, _ = bert_model(ttokens_tensor.to(device), segments_tensors.to(device))
				# 		embedding = encoded_layers[11][0]
				# 		self.premise_test.append(np.array(embedding.cpu()))
				# 		if(i%1000==0):
				# 			print(np.array(embedding.cpu()).shape)

				# temp = self.hypothesis_test
				# self.hypothesis_test = []
				# i = 0
				# for text in temp:
				# 	if(i%1000==0):
				# 		print(text, "HYP")
				# 	i = i+1
				# 	marked_text = "[CLS] " + text + " [SEP]"
				# 	tokenized_text = tokenizer.tokenize(marked_text)
				# 	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
				# 	segments_ids = [1] * len(tokenized_text)
				# 	tokens_tensor = torch.tensor([indexed_tokens])
				# 	segments_tensors = torch.tensor([segments_ids])
				# 	bert_model.eval()
				# 	with torch.no_grad():
				# 		encoded_layers, _ = bert_model(tokens_tensor.to(device), segments_tensors.to(device))
				# 		embedding = encoded_layers[11][0]
				# 		self.hypothesis_test.append(np.array(embedding.cpu()))
				# 		if(i%1000==0):
				# 			print(np.array(embedding.cpu()).shape)

				file = open('./pickle/sentences_test.dat', 'wb+')
				pickle.dump((self.premise_test, self.hypothesis_test, self.label_test), file)
				file.close()
				file = open('./pickle/maxlen_test.dat', 'wb+')
				pickle.dump((self.max_len), file)
				file.close()

				print("done on test data")
		
		else:
			file = open('./pickle/maxlen_train.dat', 'rb+')
			maxlen_train = pickle.load(file)
			file.close()
			file = open('./pickle/maxlen_test.dat', 'rb+')
			maxlen_test = pickle.load(file)
			file.close()
			self.max_len = max(maxlen_train, maxlen_test) + 1
			if(self.test):
				file = open('./pickle/sentences_test.dat', 'rb+')
				self.premise_test, self.hypothesis_test, self.label_test = pickle.load(file)
				file.close()
			else:
				file = open('./pickle/sentences_train.dat', 'rb+')
				self.premise, self.hypothesis, self.label = pickle.load(file)
				file.close()	

	def __len__(self):
		return len(self.premise)

	def __getitem__(self, idx):	
		if(not self.test):
			premise, hypothesis, label = self.premise[idx], self.hypothesis[idx], self.label[idx] #torch.tensor(self.premise[idx]).type(torch.FloatTensor), torch.tensor(self.hypothesis[idx]).type(torch.FloatTensor), torch.tensor(self.label[idx]).type(torch.LongTensor)
		else:
			premise, hypothesis, label = self.premise_test[idx], self.hypothesis_test[idx], self.label_test[idx]#torch.tensor(self.premise_test[idx]).type(torch.FloatTensor), torch.tensor(self.hypothesis_test[idx]).type(torch.FloatTensor), torch.tensor(self.label_test[idx]).type(torch.LongTensor)
		marked_text = "[CLS] " + premise + " [SEP]"
		tokenized_text = tokenizer.tokenize(marked_text)
		indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
		segments_ids = [1] * len(tokenized_text)
		tokens_tensor = torch.tensor([indexed_tokens])
		segments_tensors = torch.tensor([segments_ids])
		bert_model.eval()
		with torch.no_grad():
			encoded_layers, _ = bert_model(tokens_tensor.to(device), segments_tensors.to(device))
			embedding = encoded_layers[11][0]
			premise = np.array(embedding.cpu())
		marked_text = "[CLS] " + hypothesis + " [SEP]"
		tokenized_text = tokenizer.tokenize(marked_text)
		indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
		segments_ids = [1] * len(tokenized_text)
		tokens_tensor = torch.tensor([indexed_tokens])
		segments_tensors = torch.tensor([segments_ids])
		bert_model.eval()
		with torch.no_grad():
			encoded_layers, _ = bert_model(tokens_tensor.to(device), segments_tensors.to(device))
			embedding = encoded_layers[11][0]
			hypothesis = np.array(embedding.cpu())
		premise = torch.tensor(premise).type(torch.FloatTensor)
		hypothesis = torch.tensor(hypothesis).type(torch.FloatTensor)
		label = torch.tensor(label).type(torch.LongTensor)
		premise = pad_tensor(premise, int(1.5*self.max_len), 0)
		hypothesis = pad_tensor(hypothesis, int(1.5*self.max_len), 0)
		return premise, hypothesis, label



if __name__ == '__main__':
#	SentenceDataset(load=True, test=False)
#	SentenceDataset(load=True, test=True)
	pass
