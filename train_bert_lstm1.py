import os
import gc
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import *
from dataset_bert import *

parser = argparse.ArgumentParser(description='Experiment 1 Bert Embeddings')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') 
parser.add_argument('--batch_size', default=256, type=int) 
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--preparedata', type=int, default=1)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('==> Preparing data..')

criterion = nn.CrossEntropyLoss()

print('==> Creating networks..')
lstm = LSTM1().to(device)
lstm.load_state_dict(torch.load("./weights/networkbertlstm1_train.ckpt"))
params = lstm.parameters()
optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-5)

print('==> Loading data..')
trainset = SentenceDataset()
testset = SentenceDataset(test = True)

def train_lstm1(currepoch, epoch):
    dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Epoch: %d' % currepoch)
    
    train_loss, correct, total = 0, 0, 0

    for batch_idx in range(len(dataloader)):
        premise, hypothesis, label = next(dataloader)
        if(batch_idx==len(dataloader)-1):
            continue
        premise, hypothesis, label = premise.to(device), hypothesis.to(device), label.to(device)
        optimizer.zero_grad()
        y_pred = lstm(premise, hypothesis)

        loss = criterion(y_pred, label)
        #print(y_pred, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        with open("./logs/bertlstm1_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("./logs/bertlstm1_train_acc.log", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        del premise
        del hypothesis
        del label
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(lstm.state_dict(), './weights/networkbertlstm1_train.ckpt')
        with open("./information/bertlstm1_info.txt", "w+") as f:
            f.write("{} {}".format(currepoch, batch_idx))
        print('Batch: [%d/%d], Loss: %.3f, Train Loss: %.3f , Acc: %.3f%% (%d/%d)' % (batch_idx, len(dataloader), loss.item(), train_loss/(batch_idx+1), 100.0*correct/total, correct, total), end='\r')

    torch.save(lstm.state_dict(), './checkpoints/networkbertlstm1_train_epoch_{}.ckpt'.format(currepoch + 1))
    print('\n=> Classifier Network : Epoch [{}/{}], Loss:{:.4f}'.format(currepoch+1, epoch, train_loss / len(dataloader)))

def test_lstm1(currepoch, epoch):
    dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\n=> Testing Epoch: %d' % currepoch)
    
    test_loss, correct, total = 0, 0, 0

    for batch_idx in range(len(dataloader)):
        premise, hypothesis, label = next(dataloader)
        if(batch_idx==len(dataloader)-1):
            continue
        premise, hypothesis, label = premise.to(device), hypothesis.to(device), label.to(device)
        
        y_pred = lstm(premise, hypothesis)

        loss = criterion(y_pred, label)

        test_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        with open("./logs/bertlstm1_test_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(test_loss / total))

        with open("./logs/bertlstm1_test_acc.log", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        del premise
        del hypothesis
        del label
        gc.collect()
        torch.cuda.empty_cache()
        print('Batch: [%d/%d], Loss: %.3f, Train Loss: %.3f , Acc: %.3f%% (%d/%d)' % (batch_idx, len(dataloader), loss.item(), test_loss/(batch_idx+1), 100.0*correct/total, correct, total), end='\r')

    print('\n=> Classifier Network Test: Epoch [{}/{}], Loss:{:.4f}'.format(currepoch+1, epoch, test_loss / len(dataloader)))

print('==> Training starts..')
for epoch in range(args.epochs):
    train_lstm1(epoch, args.epochs)
    test_lstm1(epoch, args.epochs)
   
