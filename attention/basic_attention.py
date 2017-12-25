#!/usr/bin/env python
#coding:utf-8
"""
  Author:  fyh
  Purpose: show attention by MNIST 
  Created: 2017年12月19日
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import tensorboardX

writer = tensorboardX.SummaryWriter(log_dir='log')

batch_size = 32*2
in_features = 7
hidden_size = 64*2
num_layers = 1
bidirectional = False
num_directions = 1 if bidirectional == False else 2
EPOCH = 10
lam = 0.0
lr = 0.001

def prepare_data():
    transformer = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./minst',download=False, transform=transformer)
    test_data = datasets.MNIST(root='./minst',train=False, transform=transformer)
    return train_data,test_data
def test_acu(model,test_data):
    x,y = test_data.test_data.type(torch.FloatTensor)/255,torch.LongTensor(test_data.test_labels)
    #mnist dont use transform on test_data!!!!!!!!
    
    testids = np.random.random_integers(0,x.size(0)-1,200)
    x = x[testids,:,:].view(200,int(28*28/in_features),in_features)
    y = y.view(-1,1)[testids,:].squeeze()
    # just select 200 test samples randomly.

    model.eval()
    if torch.cuda.is_available():
        x,y = Variable(x.cuda(), volatile = True),Variable(y.cuda(),volatile = True)
    else:
        x,y = Variable(x,volatile = True),Variable(y,volatile = True)

    y_ , _= model(x)
    label = torch.max(y_,1)[1]
    eq = (y.cpu().data.numpy() == label.cpu().data.numpy())
    acu = eq.sum()*1.0 / eq.shape[0]

    return float(acu)

def t2v(x:torch.Tensor) -> Variable:
    if torch.cuda.is_available():
        return Variable(x.cuda())
    
    return Variable(x)

def v2num(v: Variable) -> np.ndarray:
    if torch.cuda.is_available():
        return v.data.cpu().numpy()
    return v.data.numpy()

class Attention(nn.Module):
    #----------------------------------------------------------------------
    def __init__(self):
        """"""
        super(Attention,self).__init__()
        
        self.rnn = nn.LSTM(in_features,hidden_size = hidden_size ,num_layers=num_layers,batch_first=True,dropout=0.5,bidirectional=bidirectional)
        self.rnnout_for_att = nn.Linear(hidden_size,hidden_size)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2,64),
            nn.LeakyReLU(),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32,1),
            )
        
        self.fc1 = nn.Linear(hidden_size,128)
        self.fc = nn.Linear(128,10)
    
    def forward(self,x : Variable): #x is (batch,seq_len,in_features)
        batch_size,seq_len = x.size(0), x.size(1)
        #h0,c0 = self.init_hidden(batch_size)
        out,(h,c) = self.rnn(x, None) #type: Variable,Variable,Variable
                                                  #out.size: batch, seq_len, hidden_size * num_directions
                                                  #h.size: num_layers * num_directions, batch, hidden_size
    
        #out_for_att = self.rnnout_for_att(out) #size: batch, seq_len, hidden_size
        out_for_att = out
        
        h_last = h[-1,:,:].unsqueeze(1) #type: Variable
                                        #size:batch,1,hidden_size
        #h_last = h_last.expand_as(out_for_att)
        h_last = h_last.repeat(1,seq_len,1) #size:batch,seq_len,hidden_size
        
        att_in = torch.cat((h_last,out_for_att),-1) #size:batch,seq_len,hidden_size*2
        
        att_weight = self.attention(att_in)
        att_weight = F.softmax( att_weight.view(batch_size,seq_len) )
        
        att = torch.bmm(att_weight.view(batch_size,1,seq_len),out) # batch, 1, hidden_size
        
        fc_out = self.fc1(att.view(batch_size,hidden_size))
        fc_out = self.fc(F.leaky_relu(fc_out))
        
        return F.softmax(fc_out),att_weight # fc_out:batch,10; att_weight: batch_size,seq_len
      
    
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
        c0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
        return t2v(h0),t2v(c0)
        
def draw_att(filepath, test_image):
    try:
        att = Attention()
        att.load_state_dict(torch.load(filepath))
        if torch.cuda.is_available():
            att.cuda()
        
        att.eval()
        label,weights = att(t2v(test_image.view(1,-1,in_features)))
        
        ima = weights.data.cpu() #type:torch.FloatTensor
                                 #1,28*28/in_features
        ima = ima.view(-1,1,1).expand(int(28*28/in_features),in_features,1) # expand 
        ima = ima.contiguous().view(28,28)
        ima = ima.numpy()
        plt.imshow(ima)
        plt.show()
    except:
        print('no model')
    
        

if __name__ == '__main__':
    train_data, test_data = prepare_data()
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size)
    to_image = transforms.ToPILImage()
    
    
    attention = Attention()
    
    
    if torch.cuda.is_available():
        attention.cuda()
    
    Loss = nn.CrossEntropyLoss()    
    opt = torch.optim.Adam(attention.parameters(),lr=0.001)
    lr = StepLR(opt,100,gamma=1)
    #lr = ReduceLROnPlateau(opt)
    
    attention.train()
    for epoch in range(EPOCH):
        for i, [train, label] in enumerate(train_loader):
            train = t2v(train.view(-1,int(28*28/in_features),in_features))
            label = t2v(label)
            
                      
            y,_ = attention(train)
            
            loss = Loss(y,label)
            #loss = loss + lam*torch.abs(_).sum()
            #_ = _ + 1.00e-10
            _ = torch.clamp(_, min=1.0e-12)

            loss = loss + lam * (_ * torch.log(_)).sum() / _.size(0)
           
     
            opt.zero_grad()            
            loss.backward()
            
            nn.utils.clip_grad_norm(attention.parameters(), max_norm=0.8, norm_type=2)
            opt.step()
            lr.step()
            if (epoch+1)*(i+1)%100 == 0:
                acc = test_acu(attention,test_data)
                print('epoch:{}, step:{}, loss is {},test acu is {}'.format(epoch,i,v2num(loss),acc))
                attention.train()                        
                
                writer.add_scalar('{}_{}_loss'.format(lam,lr),v2num(loss),global_step=i)
                writer.add_scalar('{}_{}_acc'.format(lam,lr),acc,global_step=i)
                
                for name,p in attention.named_parameters():
                    writer.add_histogram(name,v2num(p),global_step=i)
    
     
    
    
    torch.save(attention.state_dict(), 'model.pl')
    
    test_id = np.random.randint(len(test_data[1]))
    test = test_data[test_id][0]
    
    plt.imshow(to_image(test))
    plt.show()
    
    draw_att('model.pl',test)