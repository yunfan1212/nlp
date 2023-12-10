#coding=utf8
import pandas as pd
import torch
import os
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import argparse
import json
from transformers import BertTokenizer
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from lstm_no_length_classify_adjusting.model import LstmModel
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(filename)s-%(levelname)s-%(message)s")


class MyProcessor():
    def __init__(self,file_name):
        self.file_name=file_name
    def get_labels(self,path):
        if os.path.exists(path)==True:
            labels_dict=json.load(open(path,"r",encoding="utf8"))
        else:
            self.dfs=pd.read_csv(self.file_name,sep="\t")
            labels=list(set(self.dfs.loc[:,"label"].values))
            labels_dict={k:v for v,k in enumerate(labels)}
            with open(path,"w",encoding="utf8") as f:
                json.dump(labels_dict,f)
        return labels_dict



class MyDataSet(Dataset):
    def __init__(self,file_name):
        self.dfs=pd.read_csv(file_name,sep="\t")

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, item):
        return self.dfs.loc[item,"text"],self.dfs.loc[item,"label"]

class MyCollator():
    def __init__(self,tokenizer,max_len,label2id):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.label2id=label2id
        self.padding_id=self.tokenizer.pad_token_id
        self.cls_id=self.tokenizer.cls_token_id
    def __call__(self,data):
        text,label=zip(*self.sorted_data(data))
        B=len(text)
        L=min(len(self.tokenizer.tokenize(text[0])),self.max_len)
        text_tensor=torch.ones(B,L).long()*self.padding_id
        text_=self.tokenizer(text,padding=True,return_tensors="pt")
        input_ids=text_["input_ids"]
        attention_mask=text_["attention_mask"]

        length=torch.sum(attention_mask,dim=-1)

        for i in range(B):
            len_=length[i]
            if len_>L-1:

                text_tensor[i,:L-2]=input_ids[i,:L-2]
                text_tensor[i,L-1]=self.cls_id
            else:

                text_tensor[i,:len_]=input_ids[i,:len_]
                text_tensor[i,len_]=self.cls_id
        label=torch.tensor([self.label2id[w] for w in label]).long()
        return text_tensor,label

    def sorted_data(self,data):
        paris=list(data)
        indinces=sorted(range(len(paris)),key=lambda x:len(self.tokenizer.tokenize(paris[x][0])),reverse=True)
        paris=[paris[i] for i in indinces]
        return paris



def make_loader(collator_fn,train_file,eval_file,test_file,batch_size):
    train_loader=None
    if test_file and os.path.exists(train_file):
        train_loader=DataLoader(MyDataSet(train_file),batch_size=batch_size,shuffle=True,
                                num_workers=4,collate_fn=collator_fn)
    eval_loader=None
    if eval_file and os.path.exists(eval_file):
        eval_loader=DataLoader(MyDataSet(eval_file),batch_size=batch_size,shuffle=False,
                               num_workers=4,collate_fn=collator_fn)
    test_loader=None
    if test_file and os.path.exists(test_file):
        test_loader=DataLoader(MyDataSet(test_file),batch_size=batch_size,shuffle=False,
                               num_workers=4,collate_fn=collator_fn)
    return train_loader,eval_loader,test_loader


def args_parse():
    args=argparse.ArgumentParser(description="lstm")
    args.add_argument("--batch_size",default=64)
    args.add_argument("--train_file", default="../lstm_classify_model_adjusting/resource/train.tsv")
    args.add_argument("--eval_file", default="../lstm_classify_model_adjusting/resource/val.tsv")
    args.add_argument("--test_file", default="../lstm_classify_model_adjusting/resource/test.tsv")
    args.add_argument("--label_path", default="./resource/labels_dict.json")

    args.add_argument("--max_len", default=100)
    args.add_argument("--model_name", default="1")
    args.add_argument("--log", default="./resource/log")
    args.add_argument("--lr", default=0.001)

    args.add_argument("--embed_dim", default=64)
    args.add_argument("--hidden_dim", default=128)
    args.add_argument("--num_layers", default=2)
    args.add_argument("--dropout_rate", default=0.8)
    args.add_argument("--log_step", default=10)
    args.add_argument("--eval_step", default=50)
    args.add_argument("--save_path", default="./resource/model_out")
    args.add_argument("--eval_step1", default=20)

    args.add_argument("--max_epoch", default=10)
    args.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    args.add_argument("--label_nums", default=10)



    return args.parse_args()


def main():
    args=args_parse()
    processor=MyProcessor(args.train_file)
    label2id=processor.get_labels(args.label_path)

    tokenizer=BertTokenizer.from_pretrained("../lstm_classify_model_adjusting/resource/vocab.txt")
    collator=MyCollator(tokenizer,args.max_len,label2id)
    train_loader,eval_loader,test_loader=make_loader(collator,args.train_file,args.eval_file,
                                                     args.test_file,args.batch_size)
    args.vocab_size=len(tokenizer)
    args.label_num=len(label2id)
    model=LstmModel(args)
    model.to(args.device)

    acc=train(model,args,train_loader,eval_loader)

def train(model,args,train_loader,eval_loader):
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    writer=SummaryWriter(args.log)
    step=0
    best_acc=0
    for epoch in range(args.max_epoch):
        model.train()
        for id,batch in enumerate(train_loader):
            text,label=batch
            text.to(args.device),label.to(args.device)
            hidden=model.hidden_init(text.size(0))
            logits,_=model(text,hidden)

            loss=criterion(logits.view(-1,args.label_nums),label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step+=1
            if id%args.log_step==0:
                predict=torch.max(logits,dim=-1)[1].view(label.size()).data
                correct=(predict==label.data).sum().item()
                acc=100*correct/args.batch_size
                logging.info("train model name {} epoch {} batch {} loss {} acc {}".format(
                    args.model_name,epoch,id,loss.item(),acc
                ))
                writer.add_scalar("{}_train_acc".format(args.model_name),acc,step)
                writer.add_scalar("{}_train_loss".format(args.model_name),loss.item(),step)
            if (id+1)%args.eval_step==0:
                loss,acc=valid(model,args,eval_loader)
                logging.info("eval model name {} loss {} acc {}".format(
                    args.model_name, loss, acc
                ))
                writer.add_scalar("{}_eval_acc".format(args.model_name), acc, step)
                writer.add_scalar("{}_eval_loss".format(args.model_name), loss, step)
                if acc>best_acc:
                    best_acc=acc
                    torch.save(model.state_dict(),args.save_path+"/model_{}.pt".format(args.model_name))
    return acc


def valid(model,args,eval_loader):
    model.eval()
    criterion=nn.CrossEntropyLoss()
    predict_all=np.array([],dtype=int)
    label_all=np.array([],dtype=int)
    with torch.no_grad():
        for id,batch in enumerate(eval_loader):
            if id>args.eval_step:
                break
            text,label=batch
            text.to(args.device),label.to(args.device)
            hidden=model.hidden_init(text.size(0))
            logits,_=model(text,hidden)

            loss=criterion(logits.view(-1,args.label_num),label)

            predict=torch.max(logits,dim=-1)[1].data.cpu().numpy()
            label=label.data.cpu().numpy()
            predict_all=np.append(predict_all,predict)
            label_all=np.append(label_all,label)
        acc=accuracy_score(label_all,predict_all)
        return loss.item(),acc





if __name__ == '__main__':
    main()








