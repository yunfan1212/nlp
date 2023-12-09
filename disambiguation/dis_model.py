
import torch
from torch import nn
from transformers import BertModel
import torch.nn.functional as F

class Disambiguation(nn.Module):
    def __init__(self,args):
        super(Disambiguation,self).__init__()

        self.bert=BertModel.from_pretrained(args.model_name_or_path)
        self.bert_config=self.bert.config

        out_dim=self.bert_config.hidden_size
        self.dropout=nn.Dropout(0.3)
        self.linear=nn.Linear(out_dim,args.num_tags)
        self.criterion=nn.BCEWithLogitsLoss()


    def forward(self,input_ids, attention_mask, token_type_ids, seq_label, span1_ids,span2_ids):
        bert_outputs=self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

        token_out=bert_outputs[0]        #[batch,256,768)]           #输出层
        seq_out=bert_outputs[1]            #[batch, 768)]                #池化层

        batch_out=[]
        for t_out,sp1_mask,sp2_mask,s_out in zip(token_out,span1_ids,span2_ids,seq_out):
            sp1_mask=sp1_mask==1
            span1_out=t_out[sp1_mask]

            sp2_mask=sp2_mask==1
            span2_out=t_out[sp2_mask]

            out=torch.cat([s_out.unsqueeze(0),span1_out,span2_out],dim=0).unsqueeze(0)

            out=F.adaptive_max_pool1d(out.transpose(1,2).contiguous(),output_size=1)
            out=out.squeeze(-1)
            batch_out.append(out)
        batch_out=torch.cat(batch_out,dim=0)
        logits=self.linear(batch_out)

        if seq_out is None:
            return (None,logits)
        batch_out=self.dropout(logits)
        loss=self.criterion(batch_out,seq_label.float())
        return (loss,logits)

if __name__ == '__main__':
    model=BertModel.from_pretrained("../chinese-bert-wwm-ext/")