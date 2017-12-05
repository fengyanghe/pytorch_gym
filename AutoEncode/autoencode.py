#!/usr/bin/env python
#coding:utf-8
"""
  Author:  fyh
  Purpose: autoencoder example
  Created: 2017年12月01日
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as Opt
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import matplotlib
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.optim.lr_scheduler import StepLR, MultiStepLR #adjust learning rate during training

from tqdm import trange

import tensorboardX

writer = tensorboardX.SummaryWriter(log_dir='./log')
EPOCH = 1000
KERNAL_SIZE = 3

def creat_data():
    '''
    create test data by MNIST
    return : x_normal, label. 
    x_normal: normalized x by sklearn's standard scaler
    label: 0-9 
    '''
    data = load_digits()
    x = data.data # type: np.ndarray
    label = data.target

    showID = np.random.randint(0,x.shape[0])
    print('the {}th digit number is {}'.format(showID,label[showID]))
    plt.imshow(x[showID].reshape((8,8)),cmap=matplotlib.cm.gray)
    plt.show()
    
    x_norm = x / 16.0
    #x_norm = StandardScaler().fit_transform(x_norm)
    return x_norm, label


def tsne(x):
    '''
    use sklearn.tsne to visulization
    '''    
    xx = TSNE(n_components=2).fit_transform(x)
    return xx

def plot(x,label):
    plt.scatter(x[:,0],x[:,1],c=label*1.0/label.max(), cmap=matplotlib.cm.rainbow)
    plt.show()
    

########################################################################
class Encoder(nn.Module):
    def __init__(self,feature_num):
        super(Encoder,self).__init__()
        
        self.con1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=KERNAL_SIZE, 
                             stride=1, 
                             padding=int((KERNAL_SIZE-1)/2), # (KERNAL_SIZE-1)/2 will make sure the outsize equel insize
                             dilation=1, 
                             groups=1, 
                             bias=True) #out_channels*64 (x+2*padding-kernel_size)/stride + 1 after Conv
        nn.init.normal(self.con1.weight)
        self.c1Norm = nn.BatchNorm2d(16)
        
        self.con2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=KERNAL_SIZE, 
                              stride=1, 
                              padding=int((KERNAL_SIZE-1)/2), # (KERNAL_SIZE-1)/2 will make sure the outsize equel insize
                              dilation=1, 
                              groups=1, 
                              bias=True) #32*64
        nn.init.normal(self.con2.weight)
        self.c2Norm = nn.BatchNorm2d(32)
        
        #self.pool = nn.MaxPool2d(2,return_indices=True) #32*4*4
        self.fc = nn.Linear(32*64,128)
        nn.init.normal(self.fc.weight)   
        self.fcNorm = nn.BatchNorm1d(128)
        
        self.f2 = nn.Linear(128,32)
        nn.init.normal(self.f2.weight)  
        self.fc2Norm = nn.BatchNorm1d(32)
        
        self.out = nn.Linear(32,2)
        nn.init.normal(self.out.weight)
        nn.init.constant(self.out.bias,0)        
    
    def forward(self,x):
        con = F.leaky_relu(self.c1Norm(self.con1(x)))
        con = F.leaky_relu(self.c2Norm(self.con2(con)))
        #pool,ind = self.pool(con)
        
        fc = F.leaky_relu(self.fcNorm(self.fc(con.view(-1,32*64))))
        #return F.selu(self.out(fc)),ind
        fc = F.leaky_relu(self.fc2Norm(self.f2(fc))) 
        
        return self.out(fc) # can't work if add activation function
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        
        self.decon1 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=KERNAL_SIZE, 
                             stride=1, 
                             padding=1, 
                             dilation=1, 
                             groups=1, 
                             bias=True) #64  (x-1)*stride+kernel_size after deconv
        nn.init.normal(self.decon1.weight)
        
        
        self.decon2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=KERNAL_SIZE, 
                             stride=1, 
                             padding=1, 
                             dilation=1, 
                             groups=1, 
                             bias=True) #64  (x-1)*stride+kernel_size after deconv
                                        #out_size = (x - 1) * stride - 2 * padding + kernel_size + output_padding
                
        
        nn.init.normal(self.decon2.weight)  
        self.d2norm = nn.BatchNorm2d(16)

        self.fc = nn.Linear(128,32*8*8)# 
        nn.init.normal(self.fc.weight)       
        self.fcnorm = nn.BatchNorm1d(32*8*8)
        
        self.fc2 = nn.Linear(32,128)
        nn.init.normal(self.fc2.weight)
        nn.init.constant(self.fc2.bias,0)          
        self.fc2norm = nn.BatchNorm1d(128)
        
        self.out = nn.Linear(2,32)
        nn.init.normal(self.out.weight)
        nn.init.constant(self.out.bias,0)
        self.outnorm = nn.BatchNorm1d(32)
    
    def forward(self,x):
        out = F.leaky_relu(self.outnorm(self.out(x)))
        fc = F.leaky_relu(self.fc2norm(self.fc2(out)))
        fc = F.leaky_relu(self.fcnorm( self.fc(fc) ))
        decon2 = self.d2norm(self.decon2(fc.view(-1,32,8,8)))
        decon1 = self.decon1(decon2.view(-1,16,8,8))
     
        return decon1.squeeze()


def a2t(a:np.ndarray) -> torch.FloatTensor:
    return torch.from_numpy(a).float()
def t2v(v):
    if torch.cuda.is_available():
        v = v.cuda()
    return Variable(v)

def train(x):
    
    xv = t2v(a2t(x).view(-1,1,8,8))
    encoder = Encoder(64)
    decoder = Decoder()
    if torch.cuda.is_available:
        encoder.cuda()
        decoder.cuda()
    
    Loss = nn.L1Loss()

    opt_e = Opt.Adam(encoder.parameters(),lr = 0.1)
    opt_d = Opt.Adam(decoder.parameters(),lr = 0.1)
    
    # set learning rate of the opt_e decay by 0.5 per 1000 steps.
    slr_e = StepLR(opt_e, step_size=300,gamma=0.5) 
    # set learning rate of the opt_d decay by 0.5 per 1000 steps.
    slr_d = StepLR(opt_d, step_size=300,gamma=0.5)
    
    for i in trange(EPOCH):
        #x_encode, ind = encoder(xv)
        x_encode = encoder(xv)
        #xxv = decoder(x_encode, ind)
        xxv = decoder(x_encode)
        loss = Loss(xxv,xv.squeeze()) # type: Variable
        
        opt_e.zero_grad()
        opt_d.zero_grad()
        
        loss.backward()
    
        slr_e.step() # adjust learning rate       
        opt_e.step()
        
        slr_d.step() # adjust learning rate  
        opt_d.step()
        
        if i % 10 == 0:
            print('Epoch {} loss is {}'.format(i,loss.data.cpu().numpy()))
            for name, par in encoder.named_parameters():
                writer.add_histogram('encoder'+name, par.data.cpu().numpy(),global_step=i)
            for name, par in decoder.named_parameters():
                writer.add_histogram('decoder'+name, par.data.cpu().numpy(),global_step=i) 
            writer.add_scalar('loss',loss.data.cpu().numpy(),global_step=i)
    
    
    torch.save((encoder,decoder),'autoencoder.pl')
    return encoder,decoder
    


if __name__ == '__main__':
    x,label = creat_data()
    preprocess = StandardScaler()
    x_norm = preprocess.fit_transform(x)
    x_norm = x

    
    encoder,decoder = train(x_norm)
    
    encoder,decoder = torch.load('autoencoder.pl')
    encoder.eval()
    
    xx = encoder(t2v(a2t(x_norm)).view(-1,1,8,8)).data.cpu().numpy()
    
    plot(tsne(xx), label)
    #plot(tsne(x),label)
    
    
    showID = np.random.randint(0,x.shape[0])
    print('the {}th digit number is {}'.format(showID,label[showID]))
    plt.imshow(x[showID].reshape((8,8))*16.0,cmap=matplotlib.cm.gray)
    plt.show()
    encoder.eval()
    decoder.eval()
    recon_x = decoder(encoder(t2v(a2t(x_norm[showID])).view(-1,1,8,8)))
    #recon_x = F.sigmoid(recon_x)
    recon_x = recon_x.data.cpu().numpy()
    #recon_x = preprocess.inverse_transform(recon_x.reshape(1,-1))
    plt.imshow(recon_x.reshape((8,8))*16.0,cmap=matplotlib.cm.gray)
    plt.show()    
    
    
    
