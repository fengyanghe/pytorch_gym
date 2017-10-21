#!/usr/bin/env python
#coding:utf-8
"""
  Author:  fyh --<>
  Purpose: pytorch studying
  Created: 2017年07月04日
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import introduction2 as NetModel
import sys

def draw(x,y,y_):
    d_x = cuda2numpy(x)
    d_y = cuda2numpy(y)
    d_y_ = cuda2numpy(y_)
    plt.cla()
    plt.scatter(d_x,d_y)
    plt.plot(d_x,d_y_, color='r', lw=4)
    plt.pause(0.3)

def cuda2numpy(x):
    return x.data.cpu().numpy()
    

if __name__ == '__main__':
        
    x = torch.unsqueeze(torch.linspace(-3,3,100),1)
    y = torch.sin(x) + torch.rand(100,1)
    id_method = sys.argv[1] if len(sys.argv)>1 else 0
    
    try:
        net = torch.load('model')
    except:
        if id_method == 0 :
            net = torch.nn.Sequential(
                torch.nn.Linear(1,100),
                torch.nn.ReLU(),
                torch.nn.Linear(100,1)
            ).cuda()
            
        else:
            net = NetModel.Net(1,1).cuda()
    
    print(net)

    x, y = Variable(x).cuda(), Variable(y).cuda()
    opt = torch.optim.Adam(net.parameters())
    loss = torch.nn.MSELoss(size_average=True)
    
    plt.ion()
    for i in range(100000):
        opt.zero_grad()
        y_ = net(x)
        loss_data = loss(y_,y)
        loss_data.backward()
        opt.step()
        if i % 1000 == 0:
            print('step {0} loss is {1}'.format(i,cuda2numpy(loss_data)))
            draw(x,y,y_)
    
    torch.save(net,'model')
    plt.ioff() 
    plt.show()