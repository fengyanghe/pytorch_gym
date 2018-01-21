#!/usr/bin/env python
#coding:utf-8
"""
  Author:  fyh
  Purpose: this is wgan to create digit of mnist
  Created: 2018年01月10日
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as Opt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorboardX
import scipy.misc

writer = tensorboardX.SummaryWriter('./log')


Gen_Activator = F.sigmoid

epoch = 50
BATCH = 16

def initialize_weights(net: nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()



def prepare_data():
    transformer = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='../cnn/minst',download=False, transform=transformer)
    test_data = datasets.MNIST(root='../cnn/minst',train=False, transform=transformer)
    return train_data,test_data

########################################################################
def create_Z(size:int) -> torch.FloatTensor:
    '''
    create Variable drawn from Normal distribution
    '''
    z = np.random.rand(size,32)
    return torch.from_numpy(z).float()

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(32,64) # input_size: (batch,1); out_size:(batch, 64)
        self.fc2 = nn.Linear(64,8*4*4)

        # (H - 1) * stride - 2 * padding + kernel_size + output\_padding
        self.deconv1 = nn.ConvTranspose2d(in_channels=8, out_channels = 16, 
                                          kernel_size = 3 ) # input:(8*7*7), output:(16*6*6)
        self.bn1 = nn.BatchNorm2d(16)

        self.deconv2 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2) #input:(16*9*9, 4*14*14)
        self.bn2 = nn.BatchNorm2d(4)

        self.deconv3 = nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2) #input:(4*14*14, 1*28*28)
        self.bn3 = nn.BatchNorm2d(1)   
        initialize_weights(self)


    def forward(self, z : Variable) -> Variable:
        l1 = F.leaky_relu(self.fc1(z),0.2)
        l1 = F.leaky_relu(self.fc2(l1),0.2) # type: Variable
        l1 = l1.view(-1,8,4,4)

        l1 = self.deconv1(l1)
        l1 = self.bn1(l1)
        l1 = F.leaky_relu(l1,0.2)

        l1 = self.deconv2(l1)
        l1 = self.bn2(l1)
        l1 = F.leaky_relu(l1,0.2)   

        l1 = self.deconv3(l1)
        l1 = self.bn3(l1)
        l1 = Gen_Activator(l1)

        if Gen_Activator == F.tanh:
            l1 = (l1 + 1) / 2.0

        return l1 


########################################################################
class Discrimator(nn.Module):
    def __init__(self):
        super(Discrimator,self).__init__()

        self.con1 = nn.Conv2d(1,8,5) # input(1*28*28), output(8*24*24)
        self.p1 = nn.MaxPool2d(2) # input(8*24*24), output(8*12*12)
        self.bn = nn.BatchNorm2d(8)
        self.con2 = nn.Conv2d(8,4,5) #output(4,8,8)
        self.bn2 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4*8*8,1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.out = nn.Linear(1024,1)
        initialize_weights(self)

    def forward(self,x):
        s = self.con1(x)
        s = self.p1(s)
        s = self.bn(s)
        s = F.relu(s)

        s = self.con2(s)
        s = self.bn2(s)
        s = self.fc(F.leaky_relu(s.view(-1,4*8*8),0.2))
        s = self.bn3(s)
        s = self.out(F.leaky_relu(s,0.2))
        return F.sigmoid(s)



def t2v(tensor: torch.FloatTensor):
    if torch.cuda.is_available():
        v = Variable(tensor.cuda())
    else:
        v = Variable(tensor)
    return v


def test(gen : Generator, size = 49, path = './gen1.jpg'):    
    def imsave(images , size = [7,7] , path='./gen1.jpg'):
        return scipy.misc.imsave(path , merge(images , size))

    def merge(images , size) :
        h , w = images.shape[1] , images.shape[2]
    #      img = np.zeros((h*size[0] , w*size[1] , 3)) # :channels:3
        img = np.zeros((h*size[0] , w*size[1]))
        for idx , image in enumerate(images):
            i = idx % size[0]
            j = idx // size[1]
            img[j*h:j*h +h , i*w : i*w+w ] = image

        return img    

    gen.eval()
    z = create_Z(size)
    gen_x = gen(t2v(z)) #type: torch.FloatTensor

    gen_x = gen_x.permute(0,2,3,1).data.squeeze().cpu().numpy() * 255

    imsave(gen_x, path=path)

if __name__ == '__main__':
    train_data, _ = prepare_data()
    dataloader = Data.DataLoader(train_data,batch_size=BATCH)

    try:
        generator,discrimator = torch.load('./models.plt')

    except:
        generator = Generator()
        discrimator = Discrimator()


    if torch.cuda.is_available():
        generator.cuda()
        discrimator.cuda()

    #loss = nn.BCELoss()
    d_opt = Opt.RMSprop(discrimator.parameters(), lr = 1e-4)
    g_opt = Opt.RMSprop(generator.parameters(),lr = 1e-4)

    discrimator.train()
    generator.train()    
    for ep in range(epoch):
        for i,(real_x, label) in enumerate(dataloader):
            batch_size = real_x.size(0)
            z = create_Z(batch_size)

            fake_x = generator(t2v(z)) #type: Variable
            fake_y = t2v(torch.zeros(batch_size,1))
            real_x = t2v(real_x)
            real_y = t2v(torch.ones(batch_size,1))

            # training discriminator : just updata once
            d_opt.zero_grad() 

            # because we don't update generator, we use fake_x.detach() 
            dloss = torch.mean(discrimator(fake_x.detach())) - torch.mean(discrimator(real_x))
            dloss.backward()
            d_opt.step()
            #clip the weights into (-0.01,0.01) 
            for p in discrimator.parameters(): #type: Variable
                p.data.clamp_(-0.01,0.01)

            # training generator 
            g_opt.zero_grad()
            d_opt.zero_grad()

            z = create_Z(batch_size)
            fake_x = generator(t2v(z))

            dlabel = discrimator(fake_x) 
            gloss = -torch.mean(dlabel)
            gloss.backward()
            g_opt.step()

            if (i+1) % 200 == 0:
                gloss_print = gloss.data.cpu().numpy()
                print('epoch {0} step {1}, loss is {2}'.format(ep+1, i+1, gloss_print))
                writer.add_scalar('loss',gloss_print, global_step=(ep+1)*(i+1))
                writer.add_scalar('discrimiator loss',dloss.data.cpu().numpy(), global_step=(ep+1)*(i+1)) 
                if (i+1) % 1500==0:
                    test(generator, path='./gen{}.jpg'.format(ep))
                    discrimator.train()
                    generator.train()                     

    torch.save([generator,discrimator],'./models.plt')
    test(generator)
