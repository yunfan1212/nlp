from transformers import AutoModelForSequenceClassification,BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class BertForEntityLinking(nn.Module):
    def __init__(self, args):
        super(BertForEntityLinking, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.criterion = nn.BCEWithLogitsLoss()       #二分类损失函数，输入logits=[batch,2],label=[batch,2]
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(out_dims, args.num_tags)
    def forward(self, input_ids, attention_mask, token_type_ids, seq_label, entity_ids):

        bert_outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        #  CLS的向量
        token_out = bert_outputs[0] #[batch,256,768)]           #输出层
        seq_out = bert_outputs[1] #[batch, 768)]                #池化层

        batch_out = []                              #entity_ids:[0,0,0,1,1,1,1,0,0,0]  实体位置标注为1
        for t_out, t_mask, s_out in zip(token_out, entity_ids, seq_out):     #变量每一个样本
            t_mask = t_mask == 1 #[768]
            entity_out = t_out[t_mask] #[n,768]            #获取实体向量

            out = torch.cat([entity_out, s_out.unsqueeze(0)], dim=0).unsqueeze(0) #[1,n,768]    # 实体+判别输出

            out = F.adaptive_max_pool1d(out.transpose(1,2).contiguous(), output_size=1)   #[1,768,1]
            out = out.squeeze(-1)                                                         #[1,768]
            batch_out.append(out)       #每一个实体与判别进行拼接，提取特征信息

        batch_out = torch.cat(batch_out, dim=0)    #拼接
        batch_out = self.linear(batch_out)      #分类
        if seq_label is None:
            return (None,batch_out)
        batch_out = self.dropout(batch_out)

        loss = self.criterion(batch_out, seq_label.float())         #求损失
        return (loss,batch_out)



class BertForEntityLinking1(nn.Module):
    def __init__(self, args):
        super(BertForEntityLinking1, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size

        self.embed=nn.Embedding(self.bert_config.vocab_size,self.bert_config.hidden_size)
        self.lstm=nn.LSTM(input_size=self.bert_config.hidden_size,hidden_size=self.bert_config.hidden_size,
                          num_layers=2,batch_first=True)
        self.pool=nn.Linear(in_features=256*self.bert_config.hidden_size,out_features=self.bert_config.hidden_size)

        self.criterion = nn.BCEWithLogitsLoss()       #二分类损失函数，输入logits=[batch,2],label=[batch,2]
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(out_dims, args.num_tags)
    def forward(self, input_ids, attention_mask, token_type_ids, seq_label, entity_ids):

        # bert_outputs = self.bert(
        #     input_ids = input_ids,
        #     attention_mask = attention_mask,
        #     token_type_ids = token_type_ids,
        # )
        # #  CLS的向量
        # token_out = bert_outputs[0] #[batch,256,768)]           #输出层
        # seq_out = bert_outputs[1] #[batch, 768)]                #池化层
        output=self.embed(input_ids)
        output,state=self.lstm(output)
        seq_out=output.reshape(output.shape[0],-1)
        seq_out=self.pool(seq_out)
        token_out=output

        batch_out = []                              #entity_ids:[0,0,0,1,1,1,1,0,0,0]  实体位置标注为1
        for t_out, t_mask, s_out in zip(token_out, entity_ids, seq_out):     #变量每一个样本
            t_mask = t_mask == 1 #[768]
            t_mask=t_mask.int()[:,None]
            entity_out=t_out*t_mask
            #entity_out = t_out[t_mask] #[n,768]            #获取实体向量

            out = torch.cat([entity_out, s_out.unsqueeze(0)], dim=0).unsqueeze(0) #[1,n,768]    # 实体+判别输出

            out = F.adaptive_max_pool1d(out.transpose(1,2).contiguous(), output_size=1)   #[1,768,1]
            out = out.squeeze(-1)                                                         #[1,768]
            batch_out.append(out)       #每一个实体与判别进行拼接，提取特征信息

        batch_out = torch.cat(batch_out, dim=0)    #拼接
        batch_out = self.linear(batch_out)      #分类
        if seq_label is None:
            return (None,batch_out)
        batch_out = self.dropout(batch_out)

        loss = self.criterion(batch_out, seq_label.float())         #求损失
        return (loss,batch_out)
