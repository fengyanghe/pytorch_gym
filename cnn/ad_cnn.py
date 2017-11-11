#!/usr/bin/env python
#coding:utf-8
"""
  Author:   --<>
  Purpose: 
  Created: 2017年11月04日
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch.optim as Opt
from PIL import Image
import torchvision.transforms as transforms
import visualization as vl

get_i_image = 6 # which number you want to create image 

if __name__ == '__main__':
    trained_model = torch.load('/home/yannick/workspace/pytorch_gym/cnn/best_model') # type: nn.Module
    is_cuda = torch.cuda.is_available()
    
    test_y = torch.LongTensor([get_i_image])
    train_x = torch.rand(1,1,28,28)
    #train_x = torch.zeros(1,1,28,28)
    
    for p in trained_model.parameters(): #type: Variable
        p.requires_grad = False
        
    loss = nn.NLLLoss()

    if is_cuda:
        trained_model.cuda() # model's cuda will transfer model's parameters to gpu. but the variable created by users can't be, so avoid Variable.cuda()
        train_x = Variable(train_x.cuda(), requires_grad = True)
        test_y = Variable(test_y.cuda())
    else:
        trained_model.cpu()
        train_x = Variable(train_x, requires_grad = True)
        test_y = Variable(test_y)        
    
    opt = Opt.Adam([train_x])    
    #trained_model.train()    
    for i in range(10000):
        y_ = trained_model(train_x)
        l = loss(y_,test_y) #type: Variable
        if i == 0:
            vl.make_dot(l).view()

        opt.zero_grad()
        l.backward()
        opt.step()
        if i % 100 == 0:
            print('loss is {} \n'.format(l.cpu().data.numpy()))
            print('y_ is {} \n'.format(y_.cpu().data.numpy()))
            #print('train_x grad is {}'.format(train_x.grad.cpu().data.numpy()))
    
    best_image = train_x.cpu().data
    image = transforms.ToPILImage()(best_image.squeeze(0))
    image.save('{}.jpg'.format(get_i_image))
    plt.imshow(image)
    plt.show()