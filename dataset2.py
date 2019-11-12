from torch.utils.data import Dataset
import numpy as np 
import torch
import os
import json
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel


class SentenceDataset(Dataset):
	def __init__(self, load=False, test=False):
		super(SentenceDataset, self).__init__()

		use_cuda = torch.cuda.is_available()
		device = torch.device('cuda:0' if use_cuda else 'cpu')


		bert_model = BertModel.from_pretrained('bert-base-uncased')
		bert_model = bert_model.cuda()
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
			traindata = '../snli_1.0/snli_1.0/snli_1.0_train.jsonl'
			testdata = '../snli_1.0/snli_1.0/snli_1.0_test.jsonl'

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

				temp = self.premise
				self.premise = []
				for text in temp:
					marked_text = "[CLS] " + text + " [SEP]"
					tokenized_text = tokenizer.tokenize(marked_text)
					indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
					segments_ids = [1] * len(tokenized_text)
					tokens_tensor = torch.tensor([indexed_tokens])
					segments_tensors = torch.tensor([segments_ids])
					bert_model.eval()
					with torch.no_grad():
						encoded_layers, _ = bert_model(tokens_tensor.to(device), segments_tensors.to(device))
						embedding = encoded_layers[11][0]
						self.premise.append(np.array(embedding.cpu()))


				temp = self.hypothesis
				self.hypothesis = []
				for text in temp:
					marked_text = "[CLS] " + text + " [SEP]"
					tokenized_text = tokenizer.tokenize(marked_text)
					indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
					segments_ids = [1] * len(tokenized_text)
					tokens_tensor = torch.tensor([indexed_tokens])
					segments_tensors = torch.tensor([segments_ids])
					bert_model.eval()
					with torch.no_grad():
						encoded_layers, _ = bert_model(tokens_tensor.to(device), segments_tensors.to(device))
						embedding = encoded_layers[11][0]
						self.hypothesis.append(np.array(embedding.cpu()))

				file = open('../pickle/sentences_train.dat', 'wb+')
				pickle.dump((self.premise, self.hypothesis, self.label), file)
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

				temp = self.premise_test
				self.premise_test = []
				for text in temp:
					marked_text = "[CLS] " + text + " [SEP]"
					tokenized_text = tokenizer.tokenize(marked_text)
					indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
					segments_ids = [1] * len(tokenized_text)
					tokens_tensor = torch.tensor([indexed_tokens])
					segments_tensors = torch.tensor([segments_ids])
					bert_model.eval()
					with torch.no_grad():
						encoded_layers, _ = bert_model(ttokens_tensor.to(device), segments_tensors.to(device))
						embedding = encoded_layers[11][0]
						self.premise_test.append(np.array(embedding.cpu()))

				temp = self.hypothesis_test
				self.hypothesis_test = []
				for text in temp:
					marked_text = "[CLS] " + text + " [SEP]"
					tokenized_text = tokenizer.tokenize(marked_text)
					indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
					segments_ids = [1] * len(tokenized_text)
					tokens_tensor = torch.tensor([indexed_tokens])
					segments_tensors = torch.tensor([segments_ids])
					bert_model.eval()
					with torch.no_grad():
						encoded_layers, _ = bert_model(tokens_tensor.to(device), segments_tensors.to(device))
						embedding = encoded_layers[11][0]
						self.hypothesis_test.append(np.array(embedding.cpu()))

				file = open('../pickle/sentences_test.dat', 'wb+')
				pickle.dump((self.premise_test, self.hypothesis_test, self.label_test), file)
				file.close()
				print("done on test data")
		
		else:
			if(self.test):
				file = open('../pickle/sentences_test.dat', 'rb+')
				pickle.load((self.premise_test, self.hypothesis_test, self.label_test), file)
				file.close()
			else:
				file = open('../pickle/sentences_train.dat', 'rb+')
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
	SentenceDataset(True)
