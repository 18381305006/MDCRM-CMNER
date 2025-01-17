#!/usr/bin/env python
# coding: utf-8

import os
from tqdm.notebook import tqdm, trange
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
from transformers import AdamW,get_linear_schedule_with_warmup,get_constant_schedule,get_constant_schedule_with_warmup,get_cosine_schedule_with_warmup
from transformers import  BertPreTrainedModel, BertModel, BertConfig,BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,Dataset
from scipy import stats
import unicodedata as ucd
import random
import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import rcParams
config1 = {
    "font.family":'Times New Roman',
    "font.size": 14,
}
rcParams.update(config1)

from utils import *
from models import *
from pytorchcrf import CRF


# # 1、设置参数
data_name='ccks2017'
batch_size=8
add_new_char=True  #是否扩充目标数据的生词,为True时，需要更新BERT的维度
MAX_LEN = 256 - 2  #最大句子长度
use_crf = False #是否使用CRF
bert_model = 'bert-base-chinese'
bert_model = './bert/pretrain_bert_character/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
epochs=20#peochs
learning_rate=5e-5#学习率
scheduler_module='constant'#选择学习率变化策略：warmup，constant
if_stratigy=True#是否使用优化策略
model_name='bert_mdcrm_bilstm_stratigy'


# # 2、生成数据、模型等
#生成保存数据的文件夹
if use_crf:
    result_pathname=f'result_ner_{model_name}_crf_char'
else:
    result_pathname=f'result_ner_{model_name}_char'
if os.path.exists(f'./{result_pathname}/')==False:
    os.mkdir(f'./{result_pathname}/')
    
#读取数据
train_path=f'./data/processed_data/{data_name}/train.txt'
test_path=f'./data/processed_data/{data_name}/test.txt'
labels_path=f'./data/processed_data/{data_name}/labels.txt'
char_path=f'./data/processed_data/{data_name}/all_char.csv'
train_data=read_text(train_path,'data_label')
test_data=read_text(test_path,'data_label')
label_list=read_text(labels_path,'labels_list')
char_list=pd.read_csv(char_path,index_col=0).values[:,0]#数据中的字符

#准备标签
tag2idx = {tag: idx for idx, tag in enumerate(label_list)}
idx2tag = {idx: tag for idx, tag in enumerate(label_list)}

#加载tokenizer
tokenizer,config,new_char_list=get_tokenizer(bert_model,char_list,
                                             add_new_char=add_new_char,#是否扩充目标数据的生词
                                            )
config.__dict__['number_class']=len(tag2idx)
config.__dict__['bert_model']=bert_model
config.__dict__['use_crf']=use_crf
config.__dict__['ignore_index']=tag2idx['[PAD]']#计算误差时不需要


#数据转换为输入和输出
train_text,train_label=get_pad_data(train_data,tokenizer,tag2idx,MAX_LEN = MAX_LEN)
test_text,test_label=get_pad_data(test_data,tokenizer,tag2idx,MAX_LEN = MAX_LEN)

#转换为dataset
train_dataset=Label_Dataset(train_text,train_label)
trainloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataset=Label_Dataset(test_text,test_label)
testloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

## 加载模型
#model=Bert_BiLSTM_CRF(config).to(device)
model=MyMODEL_CRF(config).to(device)
optimizer,scheduler=set_optimizer_scheduler(model,trainloader,
                                            learning_rate=learning_rate,epochs=epochs,
                                            scheduler_module=scheduler_module)

# # 3、训练模型
best_model = None
_best_val_loss = 1e18
_best_val_acc = 1e-18

stratigy=Stratigy_loss(stat_num=2,loss_num=1,max_v=5)
_=stratigy.updata_loss_w([1])#初始化权重比例

train_epochs_result,test_epochs_result,epochs_learning_rate={},{},[]
for epoch in range(1,epochs+1):
    
    model,train_true_pred_dict,train_loss=train(epoch, model, trainloader, 
                                                    optimizer, scheduler, device,tokenizer,
                                                    if_stratigy=None if if_stratigy==False else stratigy
                                                   )
    
    model, test_true_pred_dict, test_loss = validate(epoch, model, testloader, device,tokenizer)
    
    train_result,train_report,train_pred=get_result(train_true_pred_dict,idx2tag,tag2idx)
    test_result,test_report,test_pred=get_result(test_true_pred_dict,idx2tag,tag2idx)
    #记录结果
    train_result['loss']=train_loss
    test_result['loss']=test_loss
    
    print('Train Result: ',train_result)
    print('Test Result: ',test_result)
    
    if if_stratigy==True:
        loss=stratigy.updata_loss_w([test_loss])#-------------按批次优化损失值缩放系数
    
    train_epochs_result[epoch]=train_result
    test_epochs_result[epoch]=test_result
    epochs_learning_rate.append(scheduler.get_last_lr()[0])#收集学习率
    
    acc=test_result['seq_f1']#get_acc(test_true_pred_dict)
    #if test_loss < _best_val_loss and acc > _best_val_acc:
    if acc > _best_val_acc:
        #best_model = candidate_model
        _best_val_loss = test_loss
        test_report_=test_report
        _best_val_acc = acc
            
        #保存预测的结果
        np.save(f'./{result_pathname}/测试集预测与真实label.npy',test_pred)
