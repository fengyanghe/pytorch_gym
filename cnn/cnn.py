#!/usr/bin/env python
#coding:utf-8
"""
  Author:  fyh --<>
  Purpose: basic cnn example using pytorch 0.2
  Created: 2017年09月03日
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def prepare_data():
    transformer = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./minst',download=False, transform=transformer)
    test_data = datasets.MNIST(root='./minst',train=False, download=True, transform=transformer)
    return train_data,test_data

########################################################################
class cnn(nn.Module):
    def __init__(self, in_features, out_features=10):
        """Constructor"""
        super(cnn,self).__init__()
        self.fc1 = nn.Linear(in_features, out_features=128)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), stride=1, 
                            padding=0, 
                            dilation=1, groups=1, 
                            bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, 
                                 dilation=1, 
                                 return_indices=False, 
                                 ceil_mode=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc2 = nn.Linear(64*4*4, 100)
        self.fc3 = nn.Linear(100,10)
    def forward(self, x):
        conv1 = self.conv1(x)        
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(F.relu(pool1))
       # conv2 = F.batch_norm(conv2,0.5,0.1)
        pool2 = F.relu(self.pool2(conv2))
        batchsize,c,h,w= pool2.size()        
        fc2 = F.relu(self.fc2(pool2.view(batchsize,-1)))
        fc2 = F.dropout(fc2,p = 0.5)
        fc3 = self.fc3(fc2)
        return fc3

def test_acu(model,test_data):
    x,y = test_data.test_data.type(torch.FloatTensor),torch.LongTensor(test_data.test_labels)
    testids = np.random.random_integers(0,x.size(0),200)
    x = x[testids,:,:]
    y = y.view(-1,1)[testids,:].squeeze()
    # just select 200 test samples randomly.
    
    model.eval()
    if torch.cuda.is_available():
        x,y = Variable(x.cuda()),Variable(y.cuda())
    else:
        x,y = Variable(x),Variable(y)
        
    y_ = model(x.unsqueeze(1))
    label = torch.max(y_,1)[1]
    eq = torch.eq(y,label)
    acu = torch.sum(eq.type(torch.FloatTensor))*1.0/eq.size()[0]
    acu = acu.data.numpy()
    return float(acu)
    
    
        
def main():
    train_data, test_data = prepare_data()
    data_loader = DataLoader(train_data,batch_size=10)
    cnn_model = cnn(in_features=10)
    loss = nn.CrossEntropyLoss()
    op = opt.Adam(cnn_model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8, 
                  weight_decay=0)
    is_gpu = torch.cuda.is_available()
    if is_gpu:
        cnn_model = cnn_model.cuda()
   
    best_model,best_acu = cnn_model,0
   
    for _ in range(10):
        for i, d in enumerate(data_loader):
            op.zero_grad()
            if is_gpu:
                t_d = Variable(d[0].cuda())
                t_y = Variable(d[1].cuda())
            else:
                t_d = Variable(d[0])
                t_y = Variable(d[1])
            cnn_model.train(True)
            y_ = cnn_model(t_d)
    
            l = loss(y_,t_y)
    
            #print(l[0])
            l.backward()
            op.step()  
            if i % 100 == 0:
                acu = test_acu(cnn_model, test_data)
                if best_acu < acu:
                    best_acu = acu
                    best_model = cnn_model
                print('step {0} accurate is {1}'.format(i,best_acu))
    return best_acu,best_model

if __name__ == '__main__':
    for i in range(100):
        main()
        
    