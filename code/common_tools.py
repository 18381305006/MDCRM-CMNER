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
import unicodedata as ucd#统一空格符号
import random
import warnings
warnings.filterwarnings('ignore')


class Label_Dataset(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)
    def __getitem__(self,ind):
        content_=self.data[ind]
        label_=self.label[ind]
        return content_,label_
    
    
    
class Common_tools():
    """
    对于多个任务均可使用的函数工具
    """
    def __init__(self,args=None,unvalue_class=['PAD','UNK'],sentence_config_path=None):
        self.args=args
        self.unvalue_class=unvalue_class
        self.sentence_config_path=sentence_config_path
        
        self.ner_text_file='input.seq.char'  # 文本文件
        self.ner_labels_file = 'output.seq.bio'  # 标签文件
        self.ner_class_file='vocab_bio.txt'  #类别集合文件

        if self.args.task=='MLM':
            self.data_folder='./ner_data/' #数据存放的文件夹
        else:
            self.data_folder='./ner_data/' + self.args.data_name+'/' #数据存放的文件夹
        
        self.new_word_file='./ner_data/new_word_df.csv'  #数据中新发现的字符文件
        
        self.tokenizer=None#初始化
        self.new_words_list=[]
        
    def load_tokenizer(self):
        """
        加载tokenizer
        """
        if self.tokenizer is None:
            if self.sentence_config_path is None:
                self.tokenizer=BertTokenizer.from_pretrained(self.args.pretrain_model_path)
                self.config=BertConfig.from_pretrained(self.args.pretrain_model_path)
                self.token_vocab_list=list(set(list(self.tokenizer.vocab.keys())))#Tokenizer中的字符集合
            else:
                self.tokenizer=BertTokenizer.from_pretrained(self.sentence_config_path)
                self.config=BertConfig.from_pretrained(self.sentence_config_path)
                self.token_vocab_list=list(set(list(self.tokenizer.vocab.keys())))#Tokenizer中的字符集合
        
            
    def load_ner_data(self):
        """
        加载用于NER任务的训练集和测试集
        """
        #ner的训练集
        self.train_ner_data_path=self.data_folder+'train/'+self.ner_text_file
        self.train_ner_label_path=self.data_folder+'train/'+self.ner_labels_file
        #ner的训练集
        self.test_ner_data_path=self.data_folder+'dev/'+self.ner_text_file
        self.test_ner_label_path=self.data_folder+'dev/'+self.ner_labels_file
        #ner标签集合
        self.ner_label_list_path=self.data_folder+self.ner_class_file
        
        #读取数据
        self.ner_class_list=[i for i in self.read_text(self.ner_label_list_path) if i not in self.unvalue_class]
        self.train_ner_data=self.read_text(self.train_ner_data_path)
        self.train_ner_label=self.read_text(self.train_ner_label_path)
        self.test_ner_data=self.read_text(self.test_ner_data_path)
        self.test_ner_label=self.read_text(self.test_ner_label_path)
        
        #加载字符级任务的标签
        self.cedt_class_list=['O','I']
        
        #加载句子级任务标签
        self.mpt_class_list_path=self.data_folder+'all_classs_for_sentence.npy'
        if os.path.exists(self.mpt_class_list_path)==False:
            self.mpt_class_list=list(set([i.replace('B-','').replace('I-','') 
                                       for i in self.ner_class_list if i not in ['O','START','END']]))+['None']
            np.save(self.mpt_class_list_path,self.mpt_class_list)
        else:#加载标签集合
            self.mpt_class_list=list(np.load(self.mpt_class_list_path))
        

        #更新字符集合
        self.new_words_list+=self.get_new_characters(self.train_ner_data+self.test_ner_data)
        
        
        
    def load_additional_data(self):
        """
        加载无标签的领域内专业知识数据
        """
        self.add_train_file=self.data_folder+'add_train.txt'#领域无标签数据，训练集
        self.add_test_file=self.data_folder+'add_test.txt'#领域无标签数据，测试集
        
        #加载领域无标记数据
        self.train_char_data=self.read_text(self.add_train_file)
        self.test_char_data=self.read_text(self.add_test_file)
        
        #更新字符集合
        text_list=self.train_char_data+self.test_char_data
        text_list=[' '.join([ j for j in i]) for i in text_list]
        self.new_words_list+=self.get_new_characters(text_list)
        
        #加载MLM任务的标签集合
        self.mlm_class_list=self.token_vocab_list+self.new_words_list
        
        
    def load_mpt_data(self):
        from sklearn.model_selection import train_test_split

        #读取数据
        self.mpt_text_path=self.data_folder+'creatd_texts.npy'
        self.mpt_labels_path=self.data_folder+'creatd_labels.npy'
        self.mpt_class_list_path=self.data_folder+'all_classs_for_sentence.npy'

        self.data_texts=list(np.load(self.mpt_text_path))
        self.data_labels=list(np.load(self.mpt_labels_path,allow_pickle=True))          
        
        #加载标签集合
        if os.path.exists(self.mpt_class_list_path)==False:
            self.mpt_class_list=list(set('|'.join('|'.join(i) for i in self.data_labels).split('|'))-set(['None']))+['None']
            np.save(self.mpt_class_list_path,self.mpt_class_list)
        else:
            self.mpt_class_list=list(np.load(self.mpt_class_list_path))
            
        #更新字符集合
        text_list=[' '.join([ j for j in i]) for i in self.data_texts]
        self.new_words_list+=self.get_new_characters(text_list)
        self.data_texts=text_list
        
        new_data_labels=np.zeros([len(self.data_labels),len(self.mpt_class_list)])
        for bio_id in range(len(self.data_labels)):
            bio_list=self.data_labels[bio_id]
            for bio_ in bio_list:
                new_data_labels[bio_id,self.mpt_class_list.index(bio_)]=1
                    
        self.data_labels=new_data_labels.copy()
        #拆分数据集
        self.train_creat_x,self.test_creat_x,self.train_creat_y,self.test_creat_y=train_test_split(self.data_texts,self.data_labels,
                                                       train_size=0.8)
        
        
        
    def read_text(self,text_path):
        """
        读取txt文件
        """
        with open(text_path, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines
        
    def get_new_characters(self,text_list=['今 天 有 点 胃 疼']):
        """
        从读取的数据中发现Tokenizer没有的新字符，将其添加进Tokenizer中
        """
        #启动Tokenizer工具
        self.load_tokenizer()
        
        #得到输入数据的所有字符集合
        text_list_vocab=set(' '.join(text_list).split(' '))
        
        #加载发现的新字符文件
        new_words_list=[]
        if os.path.exists(self.new_word_file):
            new_words_list=list(pd.read_csv(self.new_word_file,index_col=0).values[:,0])
            
        #更新新字符库，并保存数据
        for char in text_list_vocab:
            if char not in self.token_vocab_list:
                #if u'\u4e00' <= char <=u'\u9fff' and char not in new_words_list:
                if (char not in new_words_list) and len(char)>0:
                    new_words_list.append(char)
        new_words_df=pd.DataFrame(new_words_list)
        new_words_df.to_csv(self.new_word_file,encoding='utf_8_sig')
        return [char for char in new_words_list if char not in self.new_words_list]
    
    def get_ner_dataloader(self):
        """
        加载NER任务的数据
        """
        #读取NER数据
        self.load_ner_data()
        trainloader=Label_Dataset(self.train_ner_data,self.train_ner_label)
        trainloader=DataLoader(trainloader,batch_size=self.args.batch_size,shuffle=True)
        
        testloader=Label_Dataset(self.test_ner_data,self.test_ner_label)
        testloader=DataLoader(testloader,batch_size=self.args.batch_size,shuffle=True)
        return trainloader,testloader
    
    def get_mlm_dataloader(self):
        """
        加载MLM任务的数据
        """
        self.load_additional_data()
        trainloader=Label_Dataset(self.train_char_data,[[] for _ in self.train_char_data])
        trainloader=DataLoader(trainloader,batch_size=self.args.batch_size,shuffle=True)
        
        testloader=Label_Dataset(self.test_char_data,[[] for _ in self.test_char_data])
        testloader=DataLoader(testloader,batch_size=self.args.batch_size,shuffle=True)
        return trainloader,testloader
    
    def get_mpt_dataloader(self):
        """
        加载MPT任务的数据
        """
        #读取NER数据
        self.load_mpt_data()
        trainloader=Label_Dataset(self.train_creat_x,self.train_creat_y)
        trainloader=DataLoader(trainloader,batch_size=self.args.batch_size,shuffle=True)
        
        testloader=Label_Dataset(self.test_creat_x,self.test_creat_y)
        testloader=DataLoader(testloader,batch_size=self.args.batch_size,shuffle=True)
        return trainloader,testloader
    
    
    def upgrade_tokenizer(self):
        """
        将新字符添加进Tokenizer中
        """
        #添加新字符，返回添加的数量
        self.num_added_toks=self.tokenizer.add_tokens(self.new_words_list)
        print(self.config.vocab_size)
        self.config.vocab_size=len(self.tokenizer)
        print(self.config.vocab_size)
        #self.tokenizer.save_pretrained(self.args.repretrain_model_path)#保存token
        
        

