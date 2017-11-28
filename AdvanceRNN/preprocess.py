#!/usr/bin/env python
#coding:utf-8
"""
  Author:  yh feng
  Purpose: preprocess poetry into the lables
  Created: 2017年11月12日
  
  You should just use poetry_data, word_dict and idx2poetry(poetry_data[0])
  
  poetry_data is the list of the poetry, eg., [poetry1,poetry2] 
                 in which poetry1=[0,1,3,4,...,2],0:start, 2:end
                 
  word_dict is the word dictionary in which key is the id, value is character,
            word_dict.token2id is the inverse word dictionary in which key is the character, value is id,
            there are two special character: start and end.

  idx2poetry is the function which convert idx into a poetry.
                 eg., [0,1,3,4,...,2] -> '衆妙出洞真，煥爛曜太清。'
  
  
  the parseRawData is from the www.github.com/justdark/pytorch-poety-gen
  the poetry dataset is from www.github.com/jackeyGao/chinese-poetry
"""
import sys
import os
import json
import re

import gensim
from gensim.corpora import Dictionary

import pickle

WORD_DICT = 'dict.pl'


def parseRawData(author = None, constrain = None):
    rst = []

    def sentenceParse(para):
        # para = "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）好是去塵俗，煙花長一欄。"
        result, number = re.subn("（.*）", "", para)
        result, number = re.subn("（.*）", "", para)
        result, number = re.subn("{.*}", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                r += s;
        r, number = re.subn("。。", "。", r)
        return r

    def handleJson(file):
        # print file
        rst = []
        data = json.loads(open(file).read())
        for poetry in data:
            pdata = ""
            if (author!=None and poetry.get("author")!=author):
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                #sp = re.split("[，！。]".decode("utf-8"), s)
                sp = re.split("[，！。]", s)
                for tr in sp:
                    if constrain != None and len(tr) != constrain and len(tr)!=0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentenceParse(pdata)
            if pdata!="":
                rst.append(pdata)
        return rst
    # print sentenceParse("")
    data = []
    src = './poetry/'
    for filename in os.listdir(src):
        if filename.startswith("poet.tang"):
            data.extend(handleJson(src+filename))
    return data



def poetry2idx(poetry):    
    '''
    convert a poetry into idx of list.
    '衆妙出洞真，煥爛曜太清。' should be convert to [id(start),1,2,3,4,5,6,....,12,id(end)]
    '''    
    temp = [word_dict.token2id['start']]
    temp.extend( [word_dict.token2id[word] for word in poetry] )
    temp.append(word_dict.token2id['end'])
    return temp

def idx2poetry(idx):
    '''
    convert a idx of list into poetry.
    [id(start),1,2,3,4,5,6,....,12,id(end)] should be convert to '衆妙出洞真，煥爛曜太清。' 
    '''  
    if word_dict[idx[0]] == 'start' and word_dict[idx[-1]] == 'end':
        return ''.join([word_dict[i] for i in idx[1:-1]])
    else:
        return None


try:
    word_dict,poetry_data = pickle.load(open(WORD_DICT,'rb'))
except:
    
    poetry_data = parseRawData()
    
    word_dict = Dictionary([['start','end']])
    for poetry in poetry_data:
        word_dict.add_documents([ [i for i in poetry] ])
    
    poetry_data = [poetry2idx(poetry) for poetry in poetry_data]
    pickle.dump((word_dict,poetry_data), open(WORD_DICT,'wb'))

print('There are {} poems and {} characters to learn, consist of {} different words !!!'.format
      (word_dict.num_docs, word_dict.num_pos ,len(word_dict.keys()))
      )
        
if __name__ == '__main__':
    data = parseRawData()
    print(len(data))
    print(data[0])
    print(poetry_data[0])
    print(idx2poetry(poetry_data[0]))