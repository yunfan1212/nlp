
'''采用lstm模型分类+length'''
import pandas as pd
import torch
import torch.nn as nn
from lstm_classify_model_adjusting.model import LstmClassify
from torch.utils.data import DataLoader,Dataset
from transformers import BertTokenizer
import os
import logging
import argparse
import json
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(levelname)s-%(filename)s-%(message)s")



def get_labels(train_file,path):
    if os.path.exists(path):
        label_dict = json.load(open(path, "r", encoding="utf8"))
    else:
        dfs=pd.read_csv(train_file,sep="\t")
        labels = list(set(dfs["label"].values))
        label_dict = {v: k for k, v in enumerate(labels)}
        with open(path, "w", encoding="utf8") as f:
            json.dump(label_dict, f)
    return label_dict



class ClassDataSet(Dataset):
    def __init__(self,file_name):
        super(ClassDataSet,self).__init__()
        self.dfs=pd.read_csv(file_name,sep="\t")

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, item):
        return self.dfs.loc[item,"text"],self.dfs.loc[item,"label"]



class Collator():
    def __init__(self,tokenizer,max_len,label_dict):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.label_dict=label_dict
        self.padding_ids=tokenizer.pad_token_id
    def __call__(self, data):
        data=self.text_sorted(data)
        text,label=zip(*data)
        assert len(text)==len(label)
        batch_size=len(text)

        len_=min(len(text[0]),self.max_len)

        text_tensor=torch.ones(batch_size,len_).long()*self.padding_ids
        label=[self.label_dict[l] for l in label]
        label_tensor=torch.tensor(label).long()
        text_=self.tokenizer(text,padding=True,return_tensors="pt")

        input_ids=text_["input_ids"]
        attention_mask=text_["attention_mask"]

        for i in range(batch_size):
            text_tensor[i,:len_-1]=input_ids[i,:len_-1]
            text_tensor[i,len_-1] =self.tokenizer.sep_token_id

        length=torch.sum(attention_mask,dim=-1).data.cpu().numpy()
        length=[w if w<len_ else len_ for w in length]

        return text_tensor,label_tensor,length

    def text_sorted(self,data):

        pairs=list(data)
        indinces=sorted(range(len(pairs)),key=lambda x:len(self.tokenizer.tokenize(pairs[x][0])),
                                                                           reverse=True)
        pairs=[pairs[i] for i in indinces]
        return pairs


def make_loader(collator_fn,train_file,eval_file,test_file,batch_size):
    train_loader=None
    if train_file and os.path.exists(train_file):
        train_loader=DataLoader(ClassDataSet(train_file),shuffle=True,num_workers=4,
                                batch_size=batch_size,collate_fn=collator_fn)
    eval_loader=None
    if eval_file and os.path.exists(eval_file):
        eval_loader=DataLoader(ClassDataSet(eval_file),shuffle=False,num_workers=4,
                               batch_size=4,collate_fn=collator_fn)
    test_loader=None
    if test_file and os.path.exists(test_file):
        test_loader=DataLoader(ClassDataSet(test_file),shuffle=False,num_workers=4,
                               batch_size=batch_size,collate_fn=collator_fn)
    return train_loader,eval_loader,test_loader



def args_parse():
    args=argparse.ArgumentParser("classify")
    args.add_argument("--batch_size",default=64)
    args.add_argument("--train_file",default="./resource/train.tsv")
    args.add_argument("--eval_file", default="./resource/val.tsv")
    args.add_argument("--test_file", default="./resource/test.tsv")
    args.add_argument("--label_path",default="./resource/labels_dict.json")

    args.add_argument("--max_len", default=100)
    args.add_argument("--model_name", default="1")
    args.add_argument("--log", default="./resource/log")
    args.add_argument("--lr", default=0.001)

    args.add_argument("--embed_dim", default=64)
    args.add_argument("--hidden_dim", default=128)
    args.add_argument("--num_layers", default=2)
    args.add_argument("--drop_out_rate", default=0.8)
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
    tokenizer=BertTokenizer("./resource/vocab.txt")
    label_dict=get_labels(args.train_file,args.label_path)
    args.vocab_size=len(tokenizer)
    args.label_nums=len(label_dict)
    logging.info("load datasets")
    collator_fn=Collator(tokenizer,args.max_len,label_dict)
    train_loader,eval_loader,test_loader=make_loader(collator_fn,args.train_file,
                                                     args.eval_file,args.test_file,args.batch_size)
    logging.info("train len {} eval len {} test len {}".format(len(train_loader),len(eval_loader),len(train_loader)))
    model=LstmClassify(args)
    logging.info("load models.")

    acc=train(model,args,train_loader,eval_loader)
    return acc

def train(model,args,train_loader,eval_loader,model_name="1"):
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)

    writer=SummaryWriter(args.log)
    best_acc=0
    step=0
    for epoch in range(args.max_epoch):
        model.train()
        for id,batch in enumerate(train_loader):
            text,label,length=batch
            text.to(args.device),label.to(args.device)
            batch_size=text.size(0)

            hidden=model.hidden_init(batch_size)
            logits,_=model(text,hidden,length)
            loss=criterion(logits.view(-1,args.label_nums),label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step+=1
            if id%args.log_step==0:
                predict=torch.max(logits,dim=-1)[1].view(label.size()).data
                correct=(predict==label.data).sum()
                acc=correct/args.batch_size
                logging.info("traing model {} epoch {} batch {} loss {} acc {}".format(
                    model_name,epoch,id,loss.item(),acc.item()
                ))
                writer.add_scalar("{}_train_acc".format(model_name),acc.item(),step)
                writer.add_scalar("{}_train_loss".format(model_name),loss.item(),step)
            if (id+1)%args.eval_step==0:
                loss,acc=valid(model,args,eval_loader)
                writer.add_scalar("{}_train_acc".format(model_name), acc, step)
                writer.add_scalar("{}_train_loss".format(model_name), loss, step)

                if best_acc<acc:
                    best_acc=acc
                    torch.save(model.state_dict(),args.save_path+"/model_{}.pt".format(model_name))
    return acc


def valid(model,args,eval_loader,model_name="1"):
    criterion=nn.CrossEntropyLoss()
    predict_all=np.array([],dtype=int)
    labels_all=np.array([],dtype=int)
    model.eval()
    with torch.no_grad():
        for id,batch in enumerate(eval_loader):
            if id>args.eval_step1:
                break
            text,labels,length=batch
            text.to(args.device)
            labels.to(args.device)

            batch_size=text.size(0)
            hidden=model.hidden_init(batch_size)
            logits,hidden=model(text,hidden,length)

            loss=criterion(logits.view(-1,args.label_nums),labels).item()

            predict=torch.max(logits,dim=-1)[1].view(labels.size()).data.cpu().numpy()
            predict_all=np.append(predict_all,predict)

            labels=labels.data.cpu().numpy()
            labels_all=np.append(labels_all,labels)
        acc=accuracy_score(labels_all,predict_all)
        logging.info("eval model name {} loss {} acc {}".format(model_name,loss,acc))
        return loss,acc





if __name__ == '__main__':
    main()





















