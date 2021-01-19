import argparse
import time
from tqdm import tqdm, trange
from utils import *
from model import GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

adj, features, labels, idx_train, idx_val, idx_test = load_data()

hidden = 16
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
cuda = True
  
model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout)
if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

def train(epoch):
    t = time.time()
    for idx in trange(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        
        if idx % 100 == 0:
            model.eval()
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))
            
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=1000)
    args = parser.parse_args()
    epoch = int(args.epoch)
    print("Train On")
    train(epoch)
    print("Test On")
    test()

if __name__=="__main__":
    main()