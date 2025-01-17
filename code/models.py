import torch
import torch.nn as nn
from transformers import BertModel
from pytorchcrf import CRF

class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, config,hidden_dim=256):
        
        super(Bert_BiLSTM_CRF, self).__init__()
        self.config = config
        self.tagset_size = config.number_class
        self.embedding_dim = config.hidden_size
        self.hidden_dim = hidden_dim
        
        #加载目标bert
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.bert.resize_token_embeddings(config.vocab_size)
        
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=2, bidirectional=True, batch_first=True)
        
        self.forward_layer=nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, self.tagset_size)
        )

        if config.use_crf==True:
            self.crf = CRF(self.tagset_size, batch_first=True)
    
    def _get_features(self, sentence,attention_mask):
        #with torch.no_grad():
        embeds, _  = self.bert(sentence, attention_mask=attention_mask)[:2]
        enc, _ = self.lstm(embeds)
        feats = self.forward_layer(enc)
        return feats
    
    def _decode_no_crf_pred(self,emissions,mask):
        pred=torch.argmax(emissions,dim=2)
        
        pred_y_list=[]
        for sample_y_id in range(len(mask)):
            mask_y=mask[sample_y_id]
            pred_y=pred[sample_y_id]
            pred_y=pred_y[mask_y].cpu().tolist()
            pred_y_list.append(pred_y)
        return pred_y_list
    
    def _decode_true_no_pad(self,true,mask):
        """
        删除掉pad信息
        """
        new_true=[]
        for i in range(len(mask)):
            t=true[i][mask[i]].cpu().tolist()
            new_true.append(t)
        return new_true


    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence,mask)
        if not is_test: # Training，return loss
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)                
                loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
                seq_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
                loss = seq_loss_fct(emissions.view(-1, self.tagset_size), tags.view(-1))
                
            true_list=self._decode_true_no_pad(tags,mask)
            return decode,true_list,loss
                
        else: # Testing，return decoding
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
            return decode
        
        
        
#-------------------------BERT-MDCRM-BILSTM-CRF------------------------------
class Channel_Attention(nn.Module):
    """
    单尺度通道注意力机制
    """
    def __init__(self,input_size=768,hidden_size=256):
        super(Channel_Attention,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        
        self.key=nn.Linear(input_size,hidden_size)
        self.value=nn.Linear(input_size,hidden_size)
        self.query=nn.Linear(input_size,hidden_size)

    def forward(self,x):
        #1
        x=x.permute(0,2,1,3)
        #2
        key_layer=self.key(x)
        value_layer=self.value(x)
        query_layer=self.query(x)
        #3
        alpha_ij=torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #4
        alpha_i=torch.mean(alpha_ij,dim=3)
        alpha_i=nn.Softmax(dim=-1)(alpha_i)#归一化
        #5
        alpha_i=alpha_i.unsqueeze(2)
        attention_input=torch.matmul(alpha_i, value_layer).squeeze(2)
        return attention_input
    
    
class CRM(nn.Module):
    def __init__(self,config,in_channels=768,out_channels=768,kernel_size=3,shout_cut=True):
        super(CRM,self).__init__()
        self.config=config
        self.kernel_size=kernel_size
        self.shout_cut=shout_cut
        self.in_channels=in_channels
        
        self.conv=nn.Conv1d(in_channels=self.in_channels,
                      out_channels=out_channels,
                      kernel_size=self.kernel_size,
                      padding='same')
        self.leakrelu=nn.ReLU()
        self.layernorm=nn.LayerNorm(out_channels)
        
        if shout_cut:
            self.linear=nn.Linear(self.config.hidden_size,out_channels)
        
    def forward(self,x):
        x1=x.permute(0,2,1)
        x1=self.conv(x1)
        x1=self.leakrelu(x1)
        x1=x1.permute(0,2,1)
        
        if self.shout_cut:
            x=self.linear(x)
            x1=x+x1

        x1=self.layernorm(x1)
        return x1
    
    
class MDCRM_channel(nn.Module):
    def __init__(self,config,
                 in_channels=768,#MDCRM输入的维度
                 out_channels=768,#MDCRM输出的维度
                 kernel_size_list=[1,2,3,4,5,6,7,8],
                 shout_cut=True,
                 conv1x1_out_channels=64,
                 attentin_hidden=256,
                 hidden_dropout_prob=0.1):
        """
        out_channels：最终转换的维度
        kernel_size_list：考虑的尺度数
        conv1x1_out_channels：逐点卷积升高的通道数
        attentin_hidden：通道卷积的输出维度：out_channels->attentin_hidden
        hidden_dropout_prob：FEED层的丢弃率
        """
        super(MDCRM_channel,self).__init__()
        self.config=config
        self.out_channels=out_channels
        self.kernel_size_list=kernel_size_list
        self.shout_cut=shout_cut
        self.in_channels=in_channels
        
        #将原始信息转换维度
        self.linear=nn.Linear(self.in_channels,out_channels)
        
        #MDCRM：in_channels->[out_channels,len(kernel_size_list)]
        self.block=nn.ModuleList([CRM(self.config,in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,shout_cut=shout_cut)
                                  for kernel_size in kernel_size_list])
        
        #逐点卷积，升高通道维度：(len(kernel_size_list)+1)->conv1x1_out_channels
        self.conv1x1=nn.Sequential(
            nn.Conv2d(in_channels=len(self.kernel_size_list)+1,
                              out_channels=conv1x1_out_channels,kernel_size=1),
            nn.ReLU()
        )        
        
        #注意力机制:[out_channels,len(kernel_size_list)]->attentin_hidden
        self.channel_att=Channel_Attention(input_size=out_channels,hidden_size=attentin_hidden)
        
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出:attentin_hidden->out_channels
        self.dense = nn.Linear(attentin_hidden, out_channels)
        self.LayerNorm = nn.LayerNorm(out_channels, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(self,x):
        """
        最终维度变化：in_channels->out_channels
        """
        x0=self.linear(x)
        
        block_out=[x0.unsqueeze(3)]#将原始信息保存输入至模型中，相当于残差处理
        for i in range(len(self.kernel_size_list)):
            block_out.append(self.block[i](x).unsqueeze(3))
        x1=torch.cat(block_out,dim=3)
        x1=x1.permute(0,3,1,2)
        
        #通道升高维度
        x1=self.conv1x1(x1)
        
        #注意力
        x1=self.channel_att(x1)
        
        #FEED+LN
        x1=self.dense(x1)
        x1=self.out_dropout(x1)
        x1=self.LayerNorm(x1+x0)
        return x1
    
    
class MyMODEL_CRF(nn.Module):
    def __init__(self, config,
                 hidden_dim=256,            #LSTM的输出维度
                 mdcrm_out_channels=512,    #MDCRM输出维度（feed forward之前的输出）
                 kernel_size_list=[1,2,3,4], #kernel_size尺寸列表
                 conv1x1_out_channels=128,
                 attentin_hidden=512,        #MDCRM中注意力机制的输出（feed forward之前的输入）
                
                ):   
        super(MyMODEL_CRF, self).__init__()
        self.config = config
        self.tagset_size = config.number_class   #实体类别
        self.embedding_dim = config.hidden_size  #BERT输出维度
        self.hidden_dim = hidden_dim
        self.mdcrm_out_channels = mdcrm_out_channels
        self.kernel_size_list=kernel_size_list
        self.conv1x1_out_channels=conv1x1_out_channels
        self.attentin_hidden=attentin_hidden
        
        #加载目标bert：->config.hidden_size
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.bert.resize_token_embeddings(config.vocab_size)
        
        #加载MDCRM：config.hidden_size->mdcrm_out_channels
        self.mdcrm=MDCRM_channel(self.config,
                                 in_channels=self.embedding_dim,
                                 out_channels=mdcrm_out_channels,
                                 kernel_size_list=kernel_size_list,
                                 shout_cut=False,
                                 conv1x1_out_channels=conv1x1_out_channels,
                                 attentin_hidden=attentin_hidden,
                                 hidden_dropout_prob=0.1)
        
        #加载BiLSTM：mdcrm_out_channels->hidden_dim
        self.lstm = nn.LSTM(mdcrm_out_channels, hidden_dim // 2, num_layers=1, 
                            bidirectional=True, batch_first=True)
        
        #加载全连接层：hidden_dim->tagset_size
        self.cedt_classifier=nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim,self.tagset_size)
        )
        
        #加载CRF
        if config.use_crf==True:
            self.crf = CRF(self.tagset_size, batch_first=True)
            
    def _get_features(self, sentence,attention_mask):
        #with torch.no_grad():
        embeds, _  = self.bert(sentence, attention_mask=attention_mask)[:2]
        
        enc = self.mdcrm(embeds)
        
        enc, _ = self.lstm(enc)
        feats = self.cedt_classifier(enc)
        return feats
    
    def _decode_no_crf_pred(self,emissions,mask):
        pred=torch.argmax(emissions,dim=2)
        
        pred_y_list=[]
        for sample_y_id in range(len(mask)):
            mask_y=mask[sample_y_id]
            pred_y=pred[sample_y_id]
            pred_y=pred_y[mask_y].cpu().tolist()
            pred_y_list.append(pred_y)
        return pred_y_list
    
    def _decode_true_no_pad(self,true,mask):
        """
        删除掉pad信息
        """
        new_true=[]
        for i in range(len(mask)):
            t=true[i][mask[i]].cpu().tolist()
            new_true.append(t)
        return new_true
    
    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence,mask)
        if not is_test: # Training，return loss
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)                
                loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
                seq_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
                loss = seq_loss_fct(emissions.view(-1, self.tagset_size), tags.view(-1))
                
            true_list=self._decode_true_no_pad(tags,mask)
            return decode,true_list,loss
                
        else: # Testing，return decoding
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
            return decode
        
        
#--------------------------------BERT-BILSTM-CRF------------------------------
class MyMODEL_CRF_without_MDCRM(nn.Module):
    def __init__(self, config,
                 hidden_dim=256,            #LSTM的输出维度
                 mdcrm_out_channels=512,    #MDCRM输出维度（feed forward之前的输出）
                 kernel_size_list=[1,2,3,4], #kernel_size尺寸列表
                 conv1x1_out_channels=128,
                 attentin_hidden=512,        #MDCRM中注意力机制的输出（feed forward之前的输入）
                
                ):   
        super(MyMODEL_CRF_without_MDCRM, self).__init__()
        self.config = config
        self.tagset_size = config.number_class   #实体类别
        self.embedding_dim = config.hidden_size  #BERT输出维度
        self.hidden_dim = hidden_dim
        self.mdcrm_out_channels = mdcrm_out_channels
        self.kernel_size_list=kernel_size_list
        self.conv1x1_out_channels=conv1x1_out_channels
        self.attentin_hidden=attentin_hidden
        
        #加载目标bert：->config.hidden_size
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.bert.resize_token_embeddings(config.vocab_size)
        
        
        #加载BiLSTM：mdcrm_out_channels->hidden_dim
        mdcrm_out_channels=config.hidden_size
        self.lstm = nn.LSTM(mdcrm_out_channels, hidden_dim // 2, num_layers=1, 
                            bidirectional=True, batch_first=True)
        
        #加载全连接层：hidden_dim->tagset_size
        self.cedt_classifier=nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim,self.tagset_size)
        )
        
        #加载CRF
        if config.use_crf==True:
            self.crf = CRF(self.tagset_size, batch_first=True)
            
    def _get_features(self, sentence,attention_mask):
        #with torch.no_grad():
        embeds, _  = self.bert(sentence, attention_mask=attention_mask)[:2]
        enc, _ = self.lstm(embeds)
        feats = self.cedt_classifier(enc)
        return feats
    
    def _decode_no_crf_pred(self,emissions,mask):
        pred=torch.argmax(emissions,dim=2)
        
        pred_y_list=[]
        for sample_y_id in range(len(mask)):
            mask_y=mask[sample_y_id]
            pred_y=pred[sample_y_id]
            pred_y=pred_y[mask_y].cpu().tolist()
            pred_y_list.append(pred_y)
        return pred_y_list
    
    def _decode_true_no_pad(self,true,mask):
        """
        删除掉pad信息
        """
        new_true=[]
        for i in range(len(mask)):
            t=true[i][mask[i]].cpu().tolist()
            new_true.append(t)
        return new_true
    
    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence,mask)
        if not is_test: # Training，return loss
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)                
                loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
                seq_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
                loss = seq_loss_fct(emissions.view(-1, self.tagset_size), tags.view(-1))
                
            true_list=self._decode_true_no_pad(tags,mask)
            return decode,true_list,loss
                
        else: # Testing，return decoding
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
            return decode
        
        
#--------------------------------BERT-MDCRM-CRF------------------------------ 
class MyMODEL_CRF_without_BiLSTM(nn.Module):
    def __init__(self, config,
                 hidden_dim=256,            #LSTM的输出维度
                 mdcrm_out_channels=512,    #MDCRM输出维度（feed forward之前的输出）
                 kernel_size_list=[1,2,3,4], #kernel_size尺寸列表
                 conv1x1_out_channels=128,
                 attentin_hidden=512,        #MDCRM中注意力机制的输出（feed forward之前的输入）
                
                ):   
        super(MyMODEL_CRF_without_BiLSTM, self).__init__()
        self.config = config
        self.tagset_size = config.number_class   #实体类别
        self.embedding_dim = config.hidden_size  #BERT输出维度
        self.hidden_dim = hidden_dim
        self.mdcrm_out_channels = mdcrm_out_channels
        self.kernel_size_list=kernel_size_list
        self.conv1x1_out_channels=conv1x1_out_channels
        self.attentin_hidden=attentin_hidden
        
        #加载目标bert：->config.hidden_size
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.bert.resize_token_embeddings(config.vocab_size)
        
        #加载MDCRM：config.hidden_size->mdcrm_out_channels
        self.mdcrm=MDCRM_channel(self.config,
                                 in_channels=self.embedding_dim,
                                 out_channels=mdcrm_out_channels,
                                 kernel_size_list=kernel_size_list,
                                 shout_cut=False,
                                 conv1x1_out_channels=conv1x1_out_channels,
                                 attentin_hidden=attentin_hidden,
                                 hidden_dropout_prob=0.1)

        #加载全连接层：hidden_dim->tagset_size
        hidden_dim=mdcrm_out_channels
        
        self.cedt_classifier=nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim,self.tagset_size)
        )
        
        #加载CRF
        if config.use_crf==True:
            self.crf = CRF(self.tagset_size, batch_first=True)
            
    def _get_features(self, sentence,attention_mask):
        #with torch.no_grad():
        embeds, _  = self.bert(sentence, attention_mask=attention_mask)[:2]
        
        enc = self.mdcrm(embeds)
        
        #enc, _ = self.lstm(enc)
        feats = self.cedt_classifier(enc)
        return feats
    
    def _decode_no_crf_pred(self,emissions,mask):
        pred=torch.argmax(emissions,dim=2)
        
        pred_y_list=[]
        for sample_y_id in range(len(mask)):
            mask_y=mask[sample_y_id]
            pred_y=pred[sample_y_id]
            pred_y=pred_y[mask_y].cpu().tolist()
            pred_y_list.append(pred_y)
        return pred_y_list
    
    def _decode_true_no_pad(self,true,mask):
        """
        删除掉pad信息
        """
        new_true=[]
        for i in range(len(mask)):
            t=true[i][mask[i]].cpu().tolist()
            new_true.append(t)
        return new_true
    
    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence,mask)
        if not is_test: # Training，return loss
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)                
                loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
                seq_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
                loss = seq_loss_fct(emissions.view(-1, self.tagset_size), tags.view(-1))
                
            true_list=self._decode_true_no_pad(tags,mask)
            return decode,true_list,loss
                
        else: # Testing，return decoding
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
            return decode
        
        
#--------------------------------BERT-CRF------------------------------ 
class MyMODEL_CRF_without_MDCRM_BiLSTM(nn.Module):
    def __init__(self, config,
                 hidden_dim=256,            #LSTM的输出维度
                 mdcrm_out_channels=512,    #MDCRM输出维度（feed forward之前的输出）
                 kernel_size_list=[1,2,3,4], #kernel_size尺寸列表
                 conv1x1_out_channels=128,
                 attentin_hidden=512,        #MDCRM中注意力机制的输出（feed forward之前的输入）
                
                ):   
        super(MyMODEL_CRF_without_MDCRM_BiLSTM, self).__init__()
        self.config = config
        self.tagset_size = config.number_class   #实体类别
        self.embedding_dim = config.hidden_size  #BERT输出维度
        self.hidden_dim = hidden_dim
        self.mdcrm_out_channels = mdcrm_out_channels
        self.kernel_size_list=kernel_size_list
        self.conv1x1_out_channels=conv1x1_out_channels
        self.attentin_hidden=attentin_hidden
        
        #加载目标bert：->config.hidden_size
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.bert.resize_token_embeddings(config.vocab_size)
        
        
        #加载全连接层：hidden_dim->tagset_size
        hidden_dim=config.hidden_size
        
        self.cedt_classifier=nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim,self.tagset_size)
        )
        
        #加载CRF
        if config.use_crf==True:
            self.crf = CRF(self.tagset_size, batch_first=True)
            
    def _get_features(self, sentence,attention_mask):
        #with torch.no_grad():
        embeds, _  = self.bert(sentence, attention_mask=attention_mask)[:2]
        feats = self.cedt_classifier(embeds)
        return feats
    
    def _decode_no_crf_pred(self,emissions,mask):
        pred=torch.argmax(emissions,dim=2)
        
        pred_y_list=[]
        for sample_y_id in range(len(mask)):
            mask_y=mask[sample_y_id]
            pred_y=pred[sample_y_id]
            pred_y=pred_y[mask_y].cpu().tolist()
            pred_y_list.append(pred_y)
        return pred_y_list
    
    def _decode_true_no_pad(self,true,mask):
        """
        删除掉pad信息
        """
        new_true=[]
        for i in range(len(mask)):
            t=true[i][mask[i]].cpu().tolist()
            new_true.append(t)
        return new_true
    
    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence,mask)
        if not is_test: # Training，return loss
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)                
                loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
                seq_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
                loss = seq_loss_fct(emissions.view(-1, self.tagset_size), tags.view(-1))
                
            true_list=self._decode_true_no_pad(tags,mask)
            return decode,true_list,loss
                
        else: # Testing，return decoding
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
            return decode
        
        
#--------------------------------BERT-CNN-BILSTM-CRF------------------------------ 
class CNN(nn.Module):
    def __init__(self,in_channels=768,out_channels=768,kernel_size=3):
        super(CNN,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.conv=nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding='same')
        self.relu=nn.ReLU()
    def forward(self,x):
        x1=x.permute(0,2,1)
        x1=self.conv(x1)
        x1=self.relu(x1)
        x1=x1.permute(0,2,1)
        return x1
    
class MyMODEL_CRF_replace_MDCRM_to_CNN(nn.Module):
    def __init__(self, config,
                 hidden_dim=256,            #LSTM的输出维度
                 mdcrm_out_channels=512,    #MDCRM输出维度（feed forward之前的输出）
                 kernel_size_list=[1,2,3,4], #kernel_size尺寸列表
                 conv1x1_out_channels=128,
                 attentin_hidden=512,        #MDCRM中注意力机制的输出（feed forward之前的输入）
                
                ):   
        super(MyMODEL_CRF_replace_MDCRM_to_CNN, self).__init__()
        self.config = config
        self.tagset_size = config.number_class   #实体类别
        self.embedding_dim = config.hidden_size  #BERT输出维度
        self.hidden_dim = hidden_dim
        self.mdcrm_out_channels = mdcrm_out_channels
        self.kernel_size_list=kernel_size_list
        self.conv1x1_out_channels=conv1x1_out_channels
        self.attentin_hidden=attentin_hidden
        
        #加载目标bert：->config.hidden_size
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.bert.resize_token_embeddings(config.vocab_size)
        
        #加载CNN：config.hidden_size->mdcrm_out_channels
        self.cnn=CNN(in_channels=self.embedding_dim,
                     out_channels=mdcrm_out_channels,
                     kernel_size=3)
        
        #加载BiLSTM：mdcrm_out_channels->hidden_dim
        self.lstm = nn.LSTM(mdcrm_out_channels, hidden_dim // 2, num_layers=1, 
                            bidirectional=True, batch_first=True)

        #加载全连接层：hidden_dim->tagset_size
        self.cedt_classifier=nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim,self.tagset_size)
        )
        
        #加载CRF
        if config.use_crf==True:
            self.crf = CRF(self.tagset_size, batch_first=True)
            
    def _get_features(self, sentence,attention_mask):
        #with torch.no_grad():
        embeds, _  = self.bert(sentence, attention_mask=attention_mask)[:2]
        
        enc = self.cnn(embeds)
        enc, _ = self.lstm(enc)
        feats = self.cedt_classifier(enc)
        return feats
    
    def _decode_no_crf_pred(self,emissions,mask):
        pred=torch.argmax(emissions,dim=2)
        
        pred_y_list=[]
        for sample_y_id in range(len(mask)):
            mask_y=mask[sample_y_id]
            pred_y=pred[sample_y_id]
            pred_y=pred_y[mask_y].cpu().tolist()
            pred_y_list.append(pred_y)
        return pred_y_list
    
    def _decode_true_no_pad(self,true,mask):
        """
        删除掉pad信息
        """
        new_true=[]
        for i in range(len(mask)):
            t=true[i][mask[i]].cpu().tolist()
            new_true.append(t)
        return new_true
    
    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence,mask)
        if not is_test: # Training，return loss
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)                
                loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
                seq_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
                loss = seq_loss_fct(emissions.view(-1, self.tagset_size), tags.view(-1))
                
            true_list=self._decode_true_no_pad(tags,mask)
            return decode,true_list,loss
                
        else: # Testing，return decoding
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
            return decode
        
        
        
class Bert_CNN_CRF(nn.Module):
    def __init__(self, config,
                 hidden_dim=256,            #LSTM的输出维度
                 mdcrm_out_channels=512,    #MDCRM输出维度（feed forward之前的输出）
                 kernel_size_list=[1,2,3,4], #kernel_size尺寸列表
                 conv1x1_out_channels=128,
                 attentin_hidden=512,        #MDCRM中注意力机制的输出（feed forward之前的输入）
                
                ):   
        super(Bert_CNN_CRF, self).__init__()
        self.config = config
        self.tagset_size = config.number_class   #实体类别
        self.embedding_dim = config.hidden_size  #BERT输出维度
        self.hidden_dim = hidden_dim
        self.mdcrm_out_channels = mdcrm_out_channels
        self.kernel_size_list=kernel_size_list
        self.conv1x1_out_channels=conv1x1_out_channels
        self.attentin_hidden=attentin_hidden
        
        #加载目标bert：->config.hidden_size
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.bert.resize_token_embeddings(config.vocab_size)
        
        #加载CNN：config.hidden_size->mdcrm_out_channels
        self.cnn=CNN(in_channels=self.embedding_dim,
                     out_channels=mdcrm_out_channels,
                     kernel_size=3)

        #加载全连接层：hidden_dim->tagset_size
        self.cedt_classifier=nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(mdcrm_out_channels,self.tagset_size)
        )
        
        #加载CRF
        if config.use_crf==True:
            self.crf = CRF(self.tagset_size, batch_first=True)
            
    def _get_features(self, sentence,attention_mask):
        #with torch.no_grad():
        embeds, _  = self.bert(sentence, attention_mask=attention_mask)[:2]
        
        enc = self.cnn(embeds)
        feats = self.cedt_classifier(enc)
        return feats
    
    def _decode_no_crf_pred(self,emissions,mask):
        pred=torch.argmax(emissions,dim=2)
        
        pred_y_list=[]
        for sample_y_id in range(len(mask)):
            mask_y=mask[sample_y_id]
            pred_y=pred[sample_y_id]
            pred_y=pred_y[mask_y].cpu().tolist()
            pred_y_list.append(pred_y)
        return pred_y_list
    
    def _decode_true_no_pad(self,true,mask):
        """
        删除掉pad信息
        """
        new_true=[]
        for i in range(len(mask)):
            t=true[i][mask[i]].cpu().tolist()
            new_true.append(t)
        return new_true
    
    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence,mask)
        if not is_test: # Training，return loss
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)                
                loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
                seq_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
                loss = seq_loss_fct(emissions.view(-1, self.tagset_size), tags.view(-1))
                
            true_list=self._decode_true_no_pad(tags,mask)
            return decode,true_list,loss
                
        else: # Testing，return decoding
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
            return decode
        
        
#-------------------------BERT-BILSTM-MDCRM-CRF------------------------------
class MyMODEL_CRF_change_MDCRM_with_BILSTM(nn.Module):
    def __init__(self, config,
                 hidden_dim=256,            #LSTM的输出维度
                 mdcrm_out_channels=512,    #MDCRM输出维度（feed forward之前的输出）
                 kernel_size_list=[1,2,3,4], #kernel_size尺寸列表
                 conv1x1_out_channels=128,
                 attentin_hidden=512,        #MDCRM中注意力机制的输出（feed forward之前的输入）
                
                ):   
        super(MyMODEL_CRF_change_MDCRM_with_BILSTM, self).__init__()
        self.config = config
        self.tagset_size = config.number_class   #实体类别
        self.embedding_dim = config.hidden_size  #BERT输出维度
        self.hidden_dim = hidden_dim
        self.mdcrm_out_channels = mdcrm_out_channels
        self.kernel_size_list=kernel_size_list
        self.conv1x1_out_channels=conv1x1_out_channels
        self.attentin_hidden=attentin_hidden
        
        #加载目标bert：->config.hidden_size
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.bert.resize_token_embeddings(config.vocab_size)
        
        #加载BiLSTM：config.hidden_size->hidden_dim
        self.lstm = nn.LSTM(config.hidden_size, hidden_dim // 2, num_layers=1, 
                            bidirectional=True, batch_first=True)
        
        #加载MDCRM：hidden_dim->mdcrm_out_channels
        self.mdcrm=MDCRM_channel(self.config,
                                 in_channels=hidden_dim,
                                 out_channels=mdcrm_out_channels,
                                 kernel_size_list=kernel_size_list,
                                 shout_cut=False,
                                 conv1x1_out_channels=conv1x1_out_channels,
                                 attentin_hidden=attentin_hidden,
                                 hidden_dropout_prob=0.1)
        
        #加载全连接层：mdcrm_out_channels->tagset_size
        self.cedt_classifier=nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(mdcrm_out_channels,self.tagset_size)
        )
        
        #加载CRF
        if config.use_crf==True:
            self.crf = CRF(self.tagset_size, batch_first=True)
            
    def _get_features(self, sentence,attention_mask):
        #with torch.no_grad():
        embeds, _  = self.bert(sentence, attention_mask=attention_mask)[:2]
        enc, _ = self.lstm(embeds)
        enc = self.mdcrm(enc)
        feats = self.cedt_classifier(enc)
        return feats
    
    def _decode_no_crf_pred(self,emissions,mask):
        pred=torch.argmax(emissions,dim=2)
        
        pred_y_list=[]
        for sample_y_id in range(len(mask)):
            mask_y=mask[sample_y_id]
            pred_y=pred[sample_y_id]
            pred_y=pred_y[mask_y].cpu().tolist()
            pred_y_list.append(pred_y)
        return pred_y_list
    
    def _decode_true_no_pad(self,true,mask):
        """
        删除掉pad信息
        """
        new_true=[]
        for i in range(len(mask)):
            t=true[i][mask[i]].cpu().tolist()
            new_true.append(t)
        return new_true
    
    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence,mask)
        if not is_test: # Training，return loss
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)                
                loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
                seq_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
                loss = seq_loss_fct(emissions.view(-1, self.tagset_size), tags.view(-1))
                
            true_list=self._decode_true_no_pad(tags,mask)
            return decode,true_list,loss
                
        else: # Testing，return decoding
            if self.config.use_crf==True:
                decode=self.crf.decode(emissions, mask)
            else:
                decode=self._decode_no_crf_pred(emissions,mask)
            return decode