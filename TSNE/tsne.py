#!/usr/bin/env python
#coding:utf-8
"""
  Author:  fyh
  Purpose: tsne dimension reduction by pytorch
  Created: 2017年11月24日
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import pairwise_distances as pairwise_dis
from sklearn.datasets import load_digits as load_digits
import os
import sys
from sklearn.manifold import TSNE
import matplotlib
from scipy.spatial.distance import squareform
from sklearn import manifold
from sklearn.preprocessing import StandardScaler

import tensorboardX 

EMBEDDING = False
STEP = 0

writer = tensorboardX.SummaryWriter(log_dir='./log')

def creat_data():
    data = load_digits()
    x = data.data
    label = data.target
    

    showID = np.random.randint(0,x.shape[0])
    print('the {}th digit number is {}'.format(showID,label[showID]))
    plt.imshow(x[showID].reshape((8,8)),cmap=matplotlib.cm.gray)
    plt.show()
    
    return x, label

def a2t(a:np.ndarray) -> torch.FloatTensor:
    return torch.from_numpy(a).float()
def t2v(v):
    if torch.cuda.is_available():
        v = v.cuda()
    return Variable(v)



def getPij(trainX: np.ndarray) ->  torch.FloatTensor:
    '''
    compute pij by pairwise
    '''
    samples_num = trainX.shape[0]
    ## ||xi - xj||^2
    #xij = -pairwise_dis(trainX, n_jobs= 4, squared=True)
    #xij = a2t(xij)
    ##p(j|i):
    #mask = 1 - torch.eye(xij.size(0)) # type: torch.LongTensor
    #xij = xij[mask.byte()].view(samples_num,-1)
    #pj_i = F.softmax(xij).data
    #pj_i = pj_i*torch.exp(xij) / (torch.exp(xij) - pj_i) 

    ##p(ij):
    #pij = (pj_i+pj_i.t())/(2*trainX.shape[0])
    
    xij = pairwise_dis(trainX, n_jobs= 4, squared=True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(xij, 30, False)
    # Convert to n x n prob array
    pij = squareform(pij)    
    
    return t2v(a2t(pij))

class tsne(nn.Module):
    def __init__(self, samples_num, f_num, d_f_n):
        super(tsne,self).__init__()
        self.embed = nn.Embedding(samples_num, d_f_n)
        self.fc = nn.Linear(f_num,128)
        nn.init.normal(self.fc.weight)
        nn.init.normal(self.fc.bias)
        
        self.bn = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128,d_f_n)
        nn.init.normal(self.fc2.weight)
        nn.init.normal(self.fc2.bias)  
        
    
    def forward(self,x):
        if EMBEDDING:
            o = F.selu(self.embed(x)) # type: Variable
        else:
            o = F.selu(self.fc(x))
            o = self.bn(o)
            o = F.selu(self.fc2(o))
        
        dis = self.pairwise(o.squeeze())
        
        qij_numerator = 1 / (1 + dis)
        qij_denumerator = qij_numerator.sum(1).view(-1,1) - 1 
        
        qij = qij_numerator / qij_denumerator.sum()
        
        global STEP
        STEP += 1
        if STEP % 10 == 0: 
            for name, p in self.named_parameters():
                writer.add_histogram(name,p.data.cpu().numpy(), global_step=STEP)
        
        
        return qij,o
    
    def pairwise(self, data):
        '''
        get pairwise distance
        data: shape is (samples_num, features_num)        
        '''
        n_obs, dim = data.size()
        xk = data.unsqueeze(0).expand(n_obs, n_obs, dim)
        xl = data.unsqueeze(1).expand(n_obs, n_obs, dim)
        dkl2 = ((xk - xl)**2.0).sum(2).squeeze()
        return dkl2    

def getLoss(trainX, model, pij):
     
    samples_num, f_num = trainX.shape
        
    if EMBEDDING == True:
        x = torch.arange(0,samples_num).view(samples_num,1)
        x = t2v(x.long())
    else:
        x = t2v( a2t(trainX) )    
    
    if STEP == 0:
        qij,o = model( x )
        writer.add_graph(model, qij)    
        
    qij,o = model( x ) # type: Variable, Variable

    # remove p(i,i) and q(i,i)
    mask = 1-torch.eye(samples_num) #type: torch.LongTensor
    mask = t2v(mask.byte())
    pij = pij[mask]
    qij = qij[mask]
    
    kl = pij * torch.log(pij/qij) 
    kl = kl.sum()
    
    return kl
    
def predict(testX, model):

    samples_num, f_num = testX.shape
    model.eval()

    if EMBEDDING == True:
        x = torch.arange(0,samples_num).view(samples_num,1)
        x = t2v(x.long())
    else:
        x = t2v( a2t(testX) )    

    qij,o = model( x ) # type: Variable, Variable

    return o.squeeze().data.cpu().numpy()    

if __name__ == '__main__':
    trainx, label = creat_data()
    
    samples_num, f_num = trainx.shape
    d_f_n = 2    
    epoch = 1000
    
    pij = getPij(trainx)
    
    trainx = StandardScaler().fit_transform(trainx)
    
    model = tsne(samples_num, f_num, d_f_n)
    if torch.cuda.is_available:
        model.cuda()
    print(model)

    
    opt = torch.optim.Adam(model.parameters(),lr=0.05)
    model.train()
    
    for i in range(epoch):
        kl = getLoss(trainx , model, pij)
        opt.zero_grad()
        kl.backward()
        opt.step()
        if i % 10 == 0:
            writer.add_scalar('kl-loss',kl.data.cpu().numpy(),i)
            print('epoch {}, loss: {}'.format(i,kl.data.cpu().numpy()))
    
    xx = predict(trainx, model)
    
    # use sklearn.manifold.TSNE
    #skt = TSNE(n_components=2)
    #xx = skt.fit_transform(trainx)
    
    plt.scatter(xx[:,0],xx[:,1],c=label * 1.0 / label.max(), cmap=matplotlib.cm.rainbow)
    plt.show()
    