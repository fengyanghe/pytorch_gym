#!/usr/bin/env python
#coding:utf-8
"""
  Author:  yanghe feng
  Purpose: rnn example, create poetry by LSTM or RNN with embeding
  Created: 2017年11月12日
"""
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as Opt
import matplotlib.pyplot as plt
import os
import sys
import visualization

import preprocess
########################################################################

EMBEDDING_DIM = 256
HIDDEN_SIZE = 256
EPOCH_NUM = 10
IS_USE_LSTM = True
IS_DRAW_NN = True
BATCH_SIZE = 20
USE_BIDIRECTION = False
LAYERS_NUM = 2
Is_Determination = False

if IS_USE_LSTM:
    TRAINED_MODEL_NAME = 'LSTM_{}Direct_{}Layers.pl'.format(2 if USE_BIDIRECTION else 1, LAYERS_NUM)
else:
    TRAINED_MODEL_NAME = 'RNN_{}Direct_{}Layers.pl'.format(2 if USE_BIDIRECTION else 1, LAYERS_NUM)
    
class PoetryCreator(nn.Module):
    #----------------------------------------------------------------------
    def __init__(self, num_words, embedding_dim = EMBEDDING_DIM, hidden_size = HIDDEN_SIZE):
        super(PoetryCreator,self).__init__()
        self.embedding = nn.Embedding(num_embeddings = num_words, embedding_dim=embedding_dim, 
                                     padding_idx=None, 
                                     max_norm=None, 
                                     norm_type=2, 
                                     scale_grad_by_freq=False, 
                                     sparse=False)
       
        
        if IS_USE_LSTM:
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, 
                              num_layers=LAYERS_NUM,batch_first = True,
                              dropout = 0.5, bidirectional = USE_BIDIRECTION)
            
        else:
            self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, 
                                  num_layers=LAYERS_NUM,batch_first = True,
                                  dropout = 0.5, bidirectional = USE_BIDIRECTION)
        
        if USE_BIDIRECTION:
            self.fc_out = nn.Linear(hidden_size*2, num_words)
        else:
            self.fc_out = nn.Linear(hidden_size*1, num_words)
    
    #----------------------------------------------------------------------
    def forward(self, x:Variable, hidden:Variable) -> (Variable,Variable):
        embeds = self.embedding(x.view(1,-1)) #type: Variable, (1, x_len, embeding_dim)
       
        if IS_USE_LSTM:
            self.lstm.flatten_parameters()
            r_out, h_last= self.lstm(embeds,hidden) #type: (Variable,Variable)
                        #r_out size: (1,x_len,hidden_size*2)    
                        #h_last: h(2 directions, 1 batch, hidden_size) and c(2 directions, 1 batch, hidden_size)
        
        else:
            self.rnn.flatten_parameters()
            r_out, h_last = self.rnn(embeds,hidden) #type: (Variable,Variable)
                                                #r_out size: (1,x_len,hidden_size*2)
        
        fc = self.fc_out(r_out.squeeze(0))
        return F.log_softmax(F.leaky_relu(fc)),h_last
        

#----------------------------------------------------------------------
def lossFunction(y_pred:Variable, y:Variable) -> Variable:
    Loss = nn.NLLLoss()
    l = Loss(y_pred,y)
    return l

#----------------------------------------------------------------------
def _showPoetry(out):
    p = out.data.cpu() #type:torch.FloatTensor
    _, w_ids = p.max(1) #type:_,torch.LongTensor
    w_ids = [int(i) for i in w_ids.numpy()]
    if preprocess.word_dict[w_ids[-1]] != 'end':
        w_ids.append(preprocess.word_dict.token2id['end'])
    if preprocess.word_dict[w_ids[0]] != 'start':
        w_ids.insert(0, preprocess.word_dict.token2id['start'])        
    
    return preprocess.idx2poetry(w_ids)    

def lossByOnePoetry(poetry:Variable, h_0, model:PoetryCreator):

    y_pred, h = model(poetry, h_0) #type: Variable, Variable  
    
    y = poetry[1:]
    l = lossFunction(y_pred[0:-1,:],y)
    
    return l


    
def trainByOnePoetry(poetry:Variable, h_0, model:PoetryCreator, opt: Opt.Adam, showPoem = False):
    opt.zero_grad()
    y_pred, h = model(poetry, h_0) #type: Variable, Variable
    
    global IS_DRAW_NN
    if not IS_DRAW_NN :
        visualization.make_dot(y_pred).view()
        IS_DRAW_NN = True    
    
    y = poetry[1:]
    l = lossFunction(y_pred[0:-1,:],y)
    l.backward()
    opt.step()
    
    if showPoem:
        print(_showPoetry(y_pred))
    
    return l.data.cpu().numpy()

def init_hidden():
    directions = 2 if USE_BIDIRECTION else 1
    
    if IS_USE_LSTM:        
        h_0 = torch.zeros(directions*LAYERS_NUM,1,HIDDEN_SIZE)
        c_0 = torch.zeros(directions*LAYERS_NUM,1,HIDDEN_SIZE)
        if torch.cuda.is_available:
            h_0 =  h_0.cuda()
            c_0 =  c_0.cuda()
        h_0 = Variable( h_0 )
        c_0 = Variable( c_0 )  
        return h_0,c_0
    else:
        h_0 = torch.zeros(directions*LAYERS_NUM,1,HIDDEN_SIZE)
        if torch.cuda.is_available:
            h_0 =  h_0.cuda()
        h_0 = Variable( h_0 ) 
        return h_0       


def traningByBatch():
    num_words = len(preprocess.word_dict.keys())
    
    try:
        model = torch.load(TRAINED_MODEL_NAME)   #type:PoetryCreator   
    except:
        print('there is no trained model')
        model = PoetryCreator(num_words)   
    
    if torch.cuda.is_available:
        model.cuda()
    
    opt = Opt.RMSprop(model.parameters())
    model.train()
    for epoch in range(EPOCH_NUM):
        l = 0
        templ = 0
        for i,onePoetry in enumerate(preprocess.poetry_data):
 
            onePoetry = torch.LongTensor(onePoetry)
            if torch.cuda.is_available:
                onePoetry = onePoetry.cuda()
            
            onePoetry = Variable(onePoetry) 
            
            h_0 = init_hidden()
            
            l += lossByOnePoetry(onePoetry,h_0, model)
            if (i+1) % BATCH_SIZE == 0:  # batchsize is 100
                opt.zero_grad()
                l = l / (i+1)
                l.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(),0.7)
                opt.step()
                templ = l.data.cpu().float()
                l = 0
            
                
            if (i+1)*(epoch+1) % 500 == 0:
                print('epoch {} step {}: loss is {}'.format(epoch+1,i,templ))
                print(_createPoem(model, startwords='妙'))
    
    
    model_file = TRAINED_MODEL_NAME
    torch.save(model, model_file)

        
def traning():
    num_words = len(preprocess.word_dict.keys())
    
    try:
        model = torch.load(TRAINED_MODEL_NAME)   #type:PoetryCreator   
    except:
        print('there is no trained model')
        model = PoetryCreator(num_words)   
    
    if torch.cuda.is_available:
        model.cuda()
    
    opt = Opt.Adam(model.parameters())
    model.train()
    for epoch in range(EPOCH_NUM):
        for i,onePoetry in enumerate(preprocess.poetry_data):
 
            onePoetry = torch.LongTensor(onePoetry)
            if torch.cuda.is_available:
                onePoetry = onePoetry.cuda()
            
            onePoetry = Variable(onePoetry) 
            
            h_0 = init_hidden()
            
            l = trainByOnePoetry(onePoetry,h_0, model, opt)
                
            if i % 1000 == 0:
                l = trainByOnePoetry(onePoetry,h_0, model, opt, showPoem=True)
                print('epoch {} step {}: loss is {}'.format(epoch+1,i,l))
                print(_createPoem(model, startwords='衆妙'))
    
    
    model_file = TRAINED_MODEL_NAME
    torch.save(model, model_file)
#----------------------------------------------------------------------
def createPoetry(filename, startwords = '衆妙', is_determination = Is_Determination):
    try:
        model = torch.load(filename)   #type:PoetryCreator   
    except:
        print('there is no trained model')
        return None
    
    model.eval()

    def soft2word(model_out):
        p = model_out.data.cpu() #type:torch.FloatTensor
        if is_determination:
            _, w_id = p.max(0) #type:_,torch.LongTensor
            w_id = int(w_id.squeeze_().numpy())
        else:
            w_id = np.random.choice(p.size(0),p = np.exp(p.numpy()))
        
        return w_id
    
    
    start = preprocess.poetry2idx(startwords)[:-1]
    poetryIDs = start
    start = torch.LongTensor(start)

    if torch.cuda.is_available:
        start = start.cuda()
 
    start = Variable(start,volatile=True)
    h = init_hidden()
    
    while True:
        out, h = model(start,h)
        outID = soft2word(out[-1])
        poetryIDs.append(outID)
        if preprocess.word_dict[outID] == 'end':
            break
        elif len(poetryIDs) > 100:
            poetryIDs.append(preprocess.word_dict.token2id['end'])
            break
        else:
            start = torch.LongTensor([outID])
            if torch.cuda.is_available:
                start = start.cuda()
            start = Variable(start)
    
    return preprocess.idx2poetry(poetryIDs)      

        
        
def _createPoem(model, startwords = '衆妙'):
 
    model.eval()
    
    def soft2word(model_out):
        p = model_out.data.cpu() #type:torch.FloatTensor
        if Is_Determination:
            _, w_id = p.max(0) #type:_,torch.LongTensor
            w_id = int(w_id.squeeze_().numpy())
        else:
            w_id = np.random.choice(p.size(0),p = np.exp(p.numpy()))
        
        return w_id
    
    
    start = preprocess.poetry2idx(startwords)[:-1]
    poetryIDs = start
    start = torch.LongTensor(start)

    if torch.cuda.is_available:
        start = start.cuda()
 
    start = Variable(start)
    h = init_hidden()
    
    while True:
        out, h = model(start,h)
        outID = soft2word(out[-1])
        poetryIDs.append(outID)
        if preprocess.word_dict[outID] == 'end':
            break
        elif len(poetryIDs) > 100:
            poetryIDs.append(preprocess.word_dict.token2id['end'])
            break
        else:
            start = torch.LongTensor([outID])
            if torch.cuda.is_available:
                start = start.cuda()
            start = Variable(start)
    
    model.train()
    return preprocess.idx2poetry(poetryIDs)      

        
        


if __name__ == '__main__':   
    #traning()
    #traningByBatch()
    #poetry = createPoetry(TRAINED_MODEL_NAME, startwords='妙', is_determination=False)
    #print(poetry)
    #poetry = createPoetry(TRAINED_MODEL_NAME, startwords='衆妙')
    #print(poetry)
    startwords = sys.argv[1]
    for i in range(int(sys.argv[2])):
        print(createPoetry(TRAINED_MODEL_NAME, startwords))