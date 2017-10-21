#!/usr/bin/env python
#coding:utf-8
"""
  Author:   fyh--<>
  Purpose: construct dense net by nn.Module
  Created: 2017年07月05日
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

########################################################################
class Net(nn.Module):
    """"""
    #----------------------------------------------------------------------
    def __init__(self,in_features,out_features):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(in_features, 100)
        self.hidden2 = nn.Sequential(
            nn.Linear(100,128),
            nn.ReLU()
        )
        self.outputlayer = nn.Linear(128,out_features)
    
    def forward(self,x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.outputlayer(x)
        return x      
        
    
