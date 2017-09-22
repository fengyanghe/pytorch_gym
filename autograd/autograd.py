#!/usr/bin/env python
#coding:utf-8
"""
  Author:  fyh 
  Purpose: 
  Created: 2017年07月22日
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

#----------------------------------------------------------------------
def main():
    x = torch.FloatTensor([1.,2.]) #  create tensor 
    
    # create tensor from numpy array
    w = torch.from_numpy(np.array([8.0,9])).type(torch.FloatTensor) # type: torch.tensor
    
    #change the tensor's type 
    b = torch.LongTensor([5]).type(torch.FloatTensor)
    
    # create the Vaiable which support auto gradient when requires_grad is true
    x = Variable(x, requires_grad = True)
    w = Variable(w, requires_grad = True)
    b = Variable(b, requires_grad = True)
    
    y = x.dot(w)+b # type: Variable
    
    # two ways to change the shape of tensor or variable
    y2 = torch.mm(torch.pow(x,2).unsqueeze(0),w.view(-1,1)) + torch.sigmoid(b)
    
    # auto gradient dy/d* 
    y.backward()
    print(x.grad,w.grad,b.grad)
    
    # if grad is not to be cleared, the gradient will be accumulated
    x.grad.data.zero_(),w.grad.data.zero_(),b.grad.data.zero_()
    y2.backward()
    return x.grad,w.grad,b.grad

if __name__ == '__main__':
    print(main())