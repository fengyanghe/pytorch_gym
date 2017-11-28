#!/usr/bin/env python
#coding:utf-8
"""
  Author:  fyh --<>
  Purpose: basic rnn example using pytorch 0.2
           using tensorboard to visualize the process of training.
  Created: 2017年09月03日
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import tensorboardX

IS_VISUALIZATION = True # using tensorboard, so make sure tensorflow installed  
INPUT_ROWS = 1 # put INPUT_ROWS rows as input  
hidden_size = 128
BATCH_SIZE = 64
USE_LSTM = True
EPOCH = 3

def prepare_data():
    transformer = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./minst',download=False, transform=transformer)
    test_data = datasets.MNIST(root='./minst',train=False, transform=transformer)
    return train_data,test_data

########################################################################
class rnn(nn.Module):
    def __init__(self, out_features=10):
        """Constructor"""
        super(rnn,self).__init__()
        if USE_LSTM:
            self.lstm = nn.LSTM(28*INPUT_ROWS, hidden_size=hidden_size, batch_first=True,
                                dropout = 0.5, bidirectional = False)
        else:
            self.rnn = nn.RNN(28*INPUT_ROWS, hidden_size=hidden_size, batch_first=True,
                              dropout = 0.5, bidirectional = False)
        
        
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128,out_features)
        
    def forward(self, x):
        self.batch_size = x.data.size(0)
        if USE_LSTM:
            out, h = self.lstm(x, None) 
        else:
            out, h = self.rnn(x, self.init_hidden())       
        
        out = out[:,-1,:].view(self.batch_size,-1)
        fc1_out = self.fc1(F.relu(out))
        fc1_out = F.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)        
        
        return F.log_softmax(fc2_out)
    
    def init_hidden(self):
        h0 = torch.zeros(1,self.batch_size,hidden_size)
        if USE_LSTM:
            c0 = torch.zeros(1,self.batch_size,hidden_size)
            if torch.cuda.is_available:
                h0 = h0.cuda()
                c0 = c0.cuda()
            return Variable(h0), Variable(c0)
        else:
            if torch.cuda.is_available:
                h0 = h0.cuda()          
            return Variable(h0)

def test_acu(model,test_data):
    x,y = test_data.test_data.type(torch.FloatTensor)/255,torch.LongTensor(test_data.test_labels)
    #mnist dont use transform on test_data!!!!!!!!
    
    testids = np.random.random_integers(0,x.size(0)-1,200)
    x = x[testids,:,:].view(200,int(28/INPUT_ROWS),-1)
    y = y.view(-1,1)[testids,:].squeeze()
    # just select 200 test samples randomly.

    model.eval()
    if torch.cuda.is_available():
        x,y = Variable(x.cuda(), volatile = True),Variable(y.cuda(),volatile = True)
    else:
        x,y = Variable(x,volatile = True),Variable(y,volatile = True)

    y_ = model(x)
    label = torch.max(y_,1)[1]
    eq = (y.cpu().data.numpy() == label.cpu().data.numpy())
    acu = eq.sum()*1.0 / eq.shape[0]
    
    model.train()
    return float(acu)



def main():
    writer = tensorboardX.SummaryWriter(log_dir='log')
    
    train_data, test_data = prepare_data()
    data_loader = DataLoader(train_data,batch_size=BATCH_SIZE)
    
    rnn_model = rnn()
    loss = nn.NLLLoss()
    op = opt.Adam(rnn_model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8, 
                  weight_decay=0)
    is_gpu = torch.cuda.is_available()
    if is_gpu:
        rnn_model = rnn_model.cuda()

    for _ in range(EPOCH):          
        for i, d in enumerate(data_loader):
            
            train_x = d[0].view(-1,int(28/INPUT_ROWS),28*INPUT_ROWS)
            if is_gpu:
                t_d = Variable(train_x.cuda())
                t_y = Variable(d[1].cuda())
            else:
                t_d = Variable(train_x)
                t_y = Variable(d[1])
            
            rnn_model.train(True)
            y_ = rnn_model(t_d)

            l = loss(y_,t_y)
            if _==0 and i == 0:
                writer.add_graph(rnn_model, l)
                
            op.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm(parameters=rnn_model.parameters(),
                                    max_norm = 0.8, norm_type=2)
            op.step()
            
            writer.add_scalar('loss', l.cpu().data.numpy(),i*(1+_))
            for name,value in rnn_model.named_parameters():
                writer.add_histogram(name, value, global_step=i*(1+_), 
                                    bins='tensorflow')
            
            if i % 100 == 0:
                acu = test_acu(rnn_model, test_data)
                print('step {0} accurate is {1}'.format(i,acu))
                writer.add_scalar('acc',acu,i*(_+1))

    writer.close()

if __name__ == '__main__':
    from rnn_mnist import rnn # Make cnn as the module's cnn but not  __main__, otherwise, other program's __main__ can't load saved model.
    main()
