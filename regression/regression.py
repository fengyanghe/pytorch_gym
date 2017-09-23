#!/usr/bin/env python
#coding:utf-8
"""
  Author:   fyh
  Purpose: pytorch for regression
  Created: 2017年09月23日
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as Opt
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.datasets as data_generator
import scipy as sp
import time

plt.ion()
plt.show()
def draw_figure(x,y, fig_type = 'scatter', c = 'r') -> None :
    if x.shape[1] > 1:
        return None
    
    if fig_type == 'scatter':
        plt.scatter(x,y, c = 'b')
    elif fig_type == "plot":
        plt.plot(x,y, c = c, lw =2.5)
    else:
        return None
    
def generate_data(n_features = 1, is_drawed = False) :
    # method 1: use sklearn generate data
    if n_features > 1:
        n_informative= 2 if n_features < 4 else 5
        x,y = data_generator.make_regression(n_samples=1000, n_features=n_features, 
                                      n_informative= n_informative, 
                                      n_targets=1, 
                                      bias=0.8, 
                                      effective_rank=None, 
                                      tail_strength=0.5, 
                                      noise=2, 
                                      shuffle=True, 
                                      coef=False, 
                                      random_state=None)
    
    # method 2: generate data
    else:
        x = np.linspace(0,1,1000)+0.001
        y = np.sin(np.array(list( map(sp.math.gamma, x))) + x) + np.random.rand(1000)
    
    if is_drawed and n_features == 1 :
        draw_figure(x, y, fig_type='scatter')
    return x,y, n_features

########################################################################
# construct the one layer net: wx+b. Acturelly, it is the activation function.
class regressor (nn.Module):
    def __init__(self, n_features):
        super(regressor,self).__init__()
        
        self.regression = nn.Linear(n_features,1)
    
    def forward(self,x):
        y = self.regression(x)
        return F.tanh(y)   

# construct multi-layers network, hidden layer extend the features from one to n_hidden
class advance_regressor(nn.Module):
    def __init__(self,n_features, n_hidden = 100):
        super(advance_regressor,self).__init__()
        
        self.h1 = nn.Linear(n_features, n_hidden)
        self.regression = nn.Linear(n_hidden, 1)
    #----------------------------------------------------------------------
    def forward(self, x):
        y = F.tanh(self.h1(x))
        y = F.tanh(self.regression(y))
        return y
        
        

is_cuda = torch.cuda.is_available() # is_cuda: bool
def main(n_epoch):
    x,y,n_features = generate_data()

    # create variable which should be torch.floatTensor not doubleTensor
    x = Variable(torch.from_numpy(x).type(torch.FloatTensor)).view(-1,n_features)
    y = Variable(torch.FloatTensor(y)).view(-1,n_features)

        
    reg = regressor(n_features)
    reg_advance = advance_regressor(n_features, n_hidden=100)
    loss_fun = nn.MSELoss()
    
    if is_cuda:
        x = x.cuda()
        y = y.cuda()
        reg.cuda()
        reg_advance.cuda()

    opt = Opt.Adam(reg.parameters())
    opt_ad = Opt.Adam(reg_advance.parameters())    
       
    for i in range(n_epoch):
        y_pred = reg(x)
        y_pred_ad = reg_advance(x)
        
        loss = loss_fun(y_pred, y)
        loss_ad = loss_fun(y_pred_ad,y)
        
        opt.zero_grad()
        opt_ad.zero_grad()
        
        loss.backward()
        loss_ad.backward()
        
        opt.step()
        opt_ad.step()
        
        print('1 layer:{}, 2 layers: {}'.format(
            loss.data.cpu().numpy(),loss_ad.data.cpu().numpy()))
        if i % 10 == 0 :
            plt.cla()
            draw_figure(x.data.cpu().numpy(), y.cpu().data.numpy())
            draw_figure(x.data.cpu().numpy(), y_pred.cpu().data.numpy(), fig_type='plot')
            draw_figure(x.data.cpu().numpy(), y_pred_ad.cpu().data.numpy(), fig_type='plot', c='g')
            plt.pause(0.01)
            
        
if __name__ == '__main__':
    main(n_epoch=10000)