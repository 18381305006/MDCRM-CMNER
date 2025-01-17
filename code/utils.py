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


def read_text(path,data='text_label'):
    """
    从文件中读取数据或标签
    """
    with open(path,'r',encoding='utf_8_sig') as f:
        lines=f.readlines()
        if data=='labels_list':
            return [i.strip() for i in lines]+['[PAD]', '[CLS]', '[SEP]']
        else:
            all_data=[]
            data_sample=[]
            for dd in lines:
                dd=dd.strip()
                if '---分隔符！！！---' in dd:
                    all_data.append(data_sample)
                    data_sample=[]
                else:
                    data_sample.append(dd)
            if len(data_sample)>0:
                all_data.append(data_sample)
            return all_data
        
        
def get_pad_data(train_data,tokenizer,tag2idx,MAX_LEN = 256 - 2):
    """
    得到pad后的数据
    """
    def encode_text(text,tokenizer):
        return tokenizer.convert_tokens_to_ids(text)

    def encode_tag(label,tag2idx):
        return [tag2idx[i] for i in label]

    data_sentence,data_tag=[],[]
    ccc=0
    for sentence in train_data:
        sent_word, sent_tag = [], []
        
        for word_id in range(len(sentence)):
            word=sentence[word_id]
            split_=word.split('\t')
            if len(split_)==2:
                char,bio=split_
                sent_word.append(char)
                sent_tag.append(bio)
            else:
                print('error',word,train_data.index(sentence),ccc+word_id)
        ccc+=len(sentence)+1
        assert len(sent_word)==len(sent_tag)
        if len(sent_word)>MAX_LEN:
            sent_word_ = ['[CLS]']+sent_word[:MAX_LEN]+['[SEP]']
            sent_tag_ = ['[CLS]']+sent_tag[:MAX_LEN]+['[SEP]']
        else:
            sent_word_ = ['[CLS]']+sent_word+['[SEP]']+['[PAD]']*(MAX_LEN-len(sent_word))
            sent_tag_ = ['[CLS]']+sent_tag+['[SEP]']+['[PAD]']*(MAX_LEN-len(sent_word))

        data_sentence.append(encode_text(sent_word_,tokenizer))
        data_tag.append(encode_tag(sent_tag_,tag2idx))
        
    return data_sentence,data_tag

def get_tokenizer(bert_model,char_list,add_new_char=False):
    """
    加载tokenizer
    """
    #bert_model = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    config=BertConfig.from_pretrained(bert_model)
    
    print(config.vocab_size)
    
    #Tokenizer中的字符集合
    token_vocab_list=list(tokenizer.vocab.keys())+list(tokenizer.added_tokens_encoder.keys())
    new_char_list=[]
    
    if add_new_char==True:
        #目标数据中新发现的词，如果要加上预训练模型的新词，需要补充
        new_char_list=list(set(char_list) - set(token_vocab_list))
        num_added_toks=tokenizer.add_tokens(new_char_list)#添加新字符，返回添加的数量
        config.vocab_size=len(tokenizer)#更新词库数
        
    print(config.vocab_size)
    return tokenizer,config,new_char_list

class Label_Dataset(Dataset):
    def __init__(self,data,label):
        self.data=torch.LongTensor(data)
        self.label=torch.LongTensor(label)
        self.attention_mask=self.data!=0#掩码，True才是有价值的，0是填充的字符
    def __len__(self):
        return len(self.data)
    def __getitem__(self,ind):
        content_=self.data[ind]
        label_=self.label[ind]
        mask_=self.attention_mask[ind]
        return content_,label_,mask_
    
    
from seqeval.metrics import precision_score, recall_score, f1_score,classification_report
def get_true_no_pad(true,mask):
    """
    删除掉pad信息
    """
    new_true=[]
    for i in range(len(mask)):
        t=true[i][mask[i]].cpu().tolist()[1:-1]
        new_true.append(t)
    return new_true
def get_acc(train_true_pred_dict):
    true=train_true_pred_dict['True']
    true=np.array([int(m) for m in  '|'.join(['|'.join([str(j) for j in i]) for i in true]).split('|')])
    pred=train_true_pred_dict['Pred']
    pred=np.array([int(m) for m in  '|'.join(['|'.join([str(j) for j in i]) for i in pred]).split('|')])
    acc=(true==pred).mean()*100
    return acc
    
def train(e, model, iterator, optimizer, scheduler, device,tokenizer,if_stratigy=None):
    
    loop=tqdm(iterator)
    model.train()
    losses = 0.0
    step = 0
    pred_list,true_list,text_list=[],[],[]
    for i, batch in enumerate(loop):
        step += 1
        x, y, z = batch
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        pred,true,loss = model(x, y, z)
        text=get_true_no_pad(x,z)
        pred_list+=pred
        true_list+=true
        text_list+=[' '.join(tokenizer.convert_ids_to_tokens(text_)) for text_ in  text]
        
        losses += loss.item()
        if if_stratigy is not None:
            loss=if_stratigy.use_weight([loss])#-----------------计算加权的损失值

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        loop.set_description(f'Epoch-{e}')
        loop.set_postfix(train_loss=losses/step)

    true_pred_dict={'True':true_list,"Pred":pred_list,'Text':text_list}
    if if_stratigy is not None:
        return model,true_pred_dict,losses/step
    else:
        return model,true_pred_dict,losses/step

def validate(e, model, iterator, device,tokenizer):
    loop=tqdm(iterator)
    
    model.eval()
    losses = 0.0
    step = 0
    pred_list,true_list,text_list=[],[],[]
    with torch.no_grad():
        for i, batch in enumerate(loop):
            step += 1
            x, y, z = batch
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            pred,true,loss = model(x, y, z)
            text=get_true_no_pad(x,z)
            pred_list+=pred
            true_list+=true
            text_list+=[' '.join(tokenizer.convert_ids_to_tokens(text_)) for text_ in  text]

            losses += loss.item()

    true_pred_dict={'True':true_list,"Pred":pred_list,'Text':text_list}
    acc=get_acc(true_pred_dict)
    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(e, losses/step, acc))
    
    return model,true_pred_dict,losses/step


def get_result(train_true_pred_dict,idx2tag,tag2idx):
    """
    计算评价指标
    """    
    def compute_metrics(preds, labels):
        """
        计算评价结果
        """
        
        assert len(preds) == len(labels)
        return {"seq_precision": precision_score(labels, preds),
                "seq_recall": recall_score(labels, preds),
                "seq_f1": f1_score(labels, preds)
               }

    true=train_true_pred_dict['True']
    pred=train_true_pred_dict['Pred']
    text=train_true_pred_dict['Text']

    epoch_pred_list=[[] for _ in range(len(pred))]
    epoch_true_list=[[] for _ in range(len(true))]

    for i in range(len(epoch_pred_list)):
        pred_list=pred[i]
        true_list=true[i]
        for j in range(len(pred_list)):
            if true_list[j] not in [tag2idx['[CLS]'],tag2idx['[SEP]'],tag2idx['[PAD]']]:
                epoch_true_list[i].append(idx2tag[true_list[j]])
                if pred_list[j] not in [tag2idx['[CLS]'],tag2idx['[SEP]'],tag2idx['[PAD]']]:
                    epoch_pred_list[i].append(idx2tag[pred_list[j]])
                else:
                    epoch_pred_list[i].append('O')
    result=compute_metrics(epoch_pred_list,epoch_true_list)
    report=classification_report(epoch_true_list,epoch_pred_list,digits=5)
    return result,report,{'True Label':epoch_true_list,"Pred Label":epoch_pred_list,'Text':text}


def set_optimizer_scheduler(model,trainloader,
                            learning_rate=1e-4,epochs=10,scheduler_module='constant'):
    """
    设置优化器和学习率
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-6)

    if scheduler_module=='warmup':
        #冷启动学习率
        total_steps = len(trainloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0.1 * total_steps, #学习率每次降低10%
                                                    num_training_steps = total_steps)
    elif scheduler_module=='constant':
        #学习率保持不变
        scheduler = get_constant_schedule(optimizer)
        
    return optimizer,scheduler


def save_csv(data_df,path):
    data_df.to_csv(path,encoding='utf_8_sig')
def save_fig(path):
    plt.savefig(path,dpi=400,bbox_inches='tight')
    
def plot_subplot_result(epochs_result_test_df,data_name='Testset'):
    for column in epochs_result_test_df.columns:
        if column=='Loss':
            continue
        plt.plot(epochs_result_test_df[column],'-*',label=column)
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title(data_name)
    plt.legend()
    
class Stratigy_loss():
    def __init__(self,stat_num=3,loss_num=2,func_name='Tanh',max_v=2):
        """
        stat_num：通过前n个step来判断当前的loss的weight;
        loss_num：需要更新几个loss
        func_name：转换函数
        max_v：转换函数的缩放系数
        """
        self.stat_num=stat_num
        self.loss_num=loss_num
        self.func_name=func_name
        self.max_v=max_v
        self.stat_loss={i:[] for i in range(loss_num)}
        self.updata_num=0
        
        self.loss_loss={}#传进来的原始损失值
        self.loss_raw_k={}#从损失值计算来的原始梯度
        self.loss_k={}#原始梯度转换后的系数
    def updata_loss_w(self,loss_list_):
        loss_list=loss_list_
        try:
            loss_list=[loss.detach().cpu().numpy()+0.0 for loss in loss_list]
        except:
            pass
        print('输入的Loss：',loss_list,end=' ')
        
        loss_k=[]#临时存放每个损失值的权重
        for loss_num in range(len(loss_list)):
            if self.updata_num==0:#初始化梯度为1
                loss_k=[-1 for _ in range(len(loss_list))]
                history_loss=[]
                self.stat_loss[loss_num]=[loss_list[loss_num]]
            else:
                #第loss_num个损失值前n个阶段的损失值
                history_loss=self.stat_loss[loss_num][-(min(self.updata_num,self.stat_num)):]
                k=(loss_list[loss_num]-np.mean(history_loss))/(np.mean(history_loss))
                loss_k.append(k)
                #将最新的损失值记录下来
                self.stat_loss[loss_num]=(self.stat_loss[loss_num]+[loss_list[loss_num]])[-(min(self.updata_num+1,self.stat_num)):]

            
        self.updata_num+=1
        print('历史Loss：',history_loss,'Loss相对变换量：',loss_k,end=' ')
        self.loss_raw_k[self.updata_num]=loss_k#斜率
        loss_k=self.transform_loss_k(loss_k)#斜率转换为系数
        print('系数：',loss_k)
        self.loss_k[self.updata_num]=list(loss_k)#计算转换后的梯度值，即损失值的权重
        self.loss_loss[self.updata_num]=loss_list#真实损失值
        
        #计算转换后的总和损失
        loss=0
        for i in range(len(loss_list_)):
            loss+=loss_k[i]*loss_list_[i]
        return loss
    
    def transform_loss_k(self,k_list):
        """
        将k转换为退火后的k
        """
        new_k_list=[]
        for k in k_list:
            t_k=self.k_to_tk(k)
            new_k_list.append(t_k)
        return new_k_list
    
    def k_to_tk(self,k):
        """
        根据相对变化率计算该值被接受的概率
        """
        #计算被接受的概率
        if k<=0:
            k_pro=abs(np.tanh(k/0.1))
        else:
            k_pro=np.exp(-k/0.05)*0.5

        print('被接受的概率：',k_pro,end=' ')
        #判断是否接受
        rand=np.random.rand()
        if k_pro>=rand:
            t_k=k_pro
        else:
            t_k=self.max_v*rand
        return t_k
    
    def sigmoid_function(self,z):
        import math
        fz = []
        for num in z:
            fz.append(1/(1 + math.exp(-num)))
        return fz
    def tanh_function(self,z):
        return self.max_v*abs(np.tanh(z))

    def use_weight(self,loss_list_):
        """
        返回加权后的总损失值
        """
        #计算加权损失值
        weighted_loss=[self.loss_k[self.updata_num][i]*loss_list_[i] for i in range(len(loss_list_))]
        #weighted_loss=loss_list_
        
        return sum(weighted_loss)