import neurai.numpy as nnp
from neurai.datasets.base import Dataset
from neurai.datasets.dataloader import DataLoader
import pandas as pd
from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

class MyDataset(Dataset):
    def __init__(self,path):
        super(MyDataset,self).__init__()
        self.dfs=pd.read_csv(path)
    def __getitem__(self, item):
        text=self.dfs.loc[item,"text"]
        seq_label=self.dfs.loc[item,"seq_label"]
        entity_name=self.dfs.loc[item,"entity_name"]
        entity_start=self.dfs.loc[item,"entity_start"]
        entity_end = self.dfs.loc[item, "entity_end"]
        return [text,seq_label,entity_name,entity_start,entity_end]
    def __len__(self):
        return len(self.dfs)


def convert_ids_fn(exampels,*args):
    _input_ids=[]
    _token_type_ids=[]
    _attention_mask=[]
    _entity_ids=[]
    _seq_label=[]
    for _exam in exampels:
        raw_text=_exam[0]
        seq_label=int(_exam[1])
        entity_name=_exam[2]
        entity_start=int(_exam[3])
        entity_end=int(_exam[4])
        text_a,text_b=raw_text.split("#;#")
        seq_final_label=[0,0]
        if seq_label==0:
            seq_final_label[0]=1
        else:
            seq_final_label[1]=1
        tokenizer_pre=text_b[:entity_start]
        tokenizer_entity=entity_name
        tokenizer_post=text_b[entity_end+1:]
        real_label_start=len(tokenizer_pre)
        real_label_end=real_label_start+len(tokenizer_entity)
        tokens_b=tokenizer_pre+tokenizer_entity+tokenizer_post
        '''标注'''
        encoder_dict=tokenizer(text_a,text_pair=tokens_b,max_length=max_len,truncation="only_first",padding='max_length')
        input_ids=encoder_dict["input_ids"]
        token_type_ids=encoder_dict["token_type_ids"]
        attention_mask=encoder_dict["attention_mask"]

        offset=token_type_ids.index(1)
        entity_ids=[0]*max_len
        start_id=offset+real_label_start
        end_id=offset+real_label_end
        if end_id>max_len:
            for i in range(start_id,max_len):
                entity_ids[i]=1
        else:
            for i in range(start_id,end_id):
                entity_ids[i]=1

        _input_ids.append(input_ids)
        _attention_mask.append(attention_mask)
        _token_type_ids.append(token_type_ids)
        _seq_label.append(seq_label)
        _entity_ids.append(entity_ids)
    results={"input_ids":nnp.array(_input_ids),"attention_mask":nnp.array(_attention_mask),"token_type_ids":nnp.array(_token_type_ids),
                        "entity_ids":nnp.array(_entity_ids),"seq_label":nnp.array(_seq_label)}
    return results























