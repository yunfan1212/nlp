import torch.nn as nn
import torch
from torch.utils.data import DataLoader,Dataset
import logging
import pandas as pd
import shutil
from  collections import Counter
import os
import numpy as np
from tensorboardX import SummaryWriter
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(levelname)s-%(filename)s-%(message)s")
import argparse
from mysqltool.mysql_pool import MysqlModel
from config import Config
from tqdm import tqdm
from sentence_classify.cnn_model import CNN_Model


class MyProcessor():
    def __init__(self):
        self.raw_data_path=os.path.join(Config.path,"sentence_classify/resource/datasets.csv")


    def counter_sentence_length(self):
        dfs=pd.read_csv(self.raw_data_path)
        seq_vocab=dict()
        for i in tqdm(range(len(dfs))):
            text=dfs.loc[i,"text"]
            text=MyProcessor.date_prepare(text)
            text_list=MyProcessor.word_cut(text)
            seq_len=len(text_list)
            seq_vocab[seq_len]=seq_vocab.get(seq_len,0)+1
        seq_vocab=sorted(seq_vocab.items(),key=lambda x:x[0],reverse=False)
        logging.info("句子长度统计-句长/句频 {}".format(seq_vocab))
        return
    #获取词典
    def make_vocab(self,vocab_path,frequence=1):
        if os.path.exists(vocab_path):
            vocab=dict()
            with open(vocab_path, "r", encoding="utf8") as f:
                for line in f:
                    line=line.strip().split("\t")
                    if len(line)==2:
                        vocab[line[0]]=int(line[1])
            return vocab
        dfs=pd.read_csv(self.raw_data_path)
        counter=Counter()
        for i in range(len(dfs)):
            text=dfs.loc[i,"text"]
            text=MyProcessor.date_prepare(text)
            text_list=MyProcessor.word_cut(text)
            for w in text_list:
                counter.update(w)
        counter=counter.most_common()
        tokens=["UNK","PAD","SOS","SEP"]+[k for k,v in counter if v>frequence]
        vocab={v:k for k,v in enumerate(tokens) }
        with open(vocab_path,"w",encoding="utf8") as f:
            for k,v in vocab.items():
                f.write("{}\t{}\n".format(k,v))
        return vocab

    @staticmethod
    def word_cut(text):
        text=["SOS"]+[w for w in list(text) if w!="" and w!=' ']+["SEP"]
        return text


    #从数据中获取数据并保存到指定文件夹中
    def get_datesets(self):
        sql = "SELECT first_category,product_desc,product_name FROM znzz.a_znzz_enterprise_product where first_category!=''"
        res = MysqlModel.query(sql)
        dfs=pd.DataFrame(res)
        dfs1=[]
        counter=dict()
        equipment=[]
        safe=[]
        automation=[]
        software=[]

        for i in tqdm(range(len(dfs))):
            text=MyProcessor.date_prepare(dfs.loc[i,"product_desc"])
            text1=MyProcessor.date_prepare(dfs.loc[i,"product_name"])
            label=dfs.loc[i,"first_category"]
            dfs.loc[i,"product_desc"]=text1+text
            if text1+text!="":
                dfs1.append({"text":text1+text,"label":label})
                if label=="智能装备":
                    equipment.append({"text":text1+text,"label":label})
                elif label=="安全":
                    safe.append({"text":text1+text,"label":label})
                elif label=="工业自动化":
                    automation.append({"text":text1+text,"label":label})
                elif label=="工业软件与平台":
                    software.append({"text":text1+text,"label":label})

                counter[label]=counter.get(label,0)+1
        dfs1=pd.DataFrame(dfs1)
        dfs1.to_csv(os.path.join(Config.path,"sentence_classify/resource/datasets.csv"))
        logging.info(counter)

        dfs2=equipment[:900]+safe[:900]+automation[:900]+software[:900]
        train_indicator=np.arange(len(dfs2))
        np.random.shuffle(train_indicator)
        train=[dfs2[i] for i in train_indicator[:int(len(train_indicator)*0.8)]]
        test=[dfs2[i] for i in train_indicator[int(len(train_indicator)*0.8):]]
        dfs2=pd.DataFrame(train)
        dfs2.to_csv(os.path.join(Config.path,"sentence_classify/resource/train.csv"))
        dfs3=pd.DataFrame(test)
        dfs3.to_csv(os.path.join(Config.path,"sentence_classify/resource/test.csv"))

    #数据预处理
    @staticmethod
    def date_prepare(text):
        if type(text)==str:
            text=text.strip()
            text=text.replace("\n","")
            text = text.replace("\t", " ")
            text=text.replace("　","")
            return text
        else:
            return ""

    def get_class_vocab(self,label_path,re_train=False):
        if re_train==True or not os.path.exists(label_path):
            sql="SELECT distinct first_category FROM znzz.a_znzz_enterprise_product where first_category!=''"
            res=MysqlModel.query(sql)
            label_list=[w["first_category"] for w in res]
            label_list.append("其他")
            label_vocab={k:v for v,k in enumerate(label_list)}
            with open(label_path,"w",encoding="utf8") as f:
                for k,v in label_vocab.items():
                    f.write("{}\t{}\n".format(k,v))
        else:
            label_vocab=dict()
            with open(label_path,"r",encoding="utf8") as f:
                for line in f:
                    line=line.strip().split("\t")
                    label_vocab[line[0]]=int(line[1])
        return label_vocab







class MyDataSet(Dataset):
    def __init__(self,path):
        self.dfs=pd.read_csv(path)
        indicator=np.arange(len(self.dfs))
        np.random.shuffle(indicator)
        self.dfs=self.dfs.loc[indicator,:]
        self.dfs.index=range(len(self.dfs))
        self.dfs=self.dfs.drop(columns=["Unnamed: 0"])

    def __getitem__(self, item):

        return self.dfs.loc[item,"text"],self.dfs.loc[item,"label"]
    def __len__(self):
        return len(self.dfs)

class MyCollator():
    def __init__(self,vocab_dict,label_dict,max_len,min_len):
        self.vocab_dict=vocab_dict
        self.pad_id=self.vocab_dict["PAD"]
        self.nuk_id=self.vocab_dict["UNK"]
        self.sos_id=self.vocab_dict["SOS"]
        self.eos_id=self.vocab_dict["SEP"]
        self.label_dict=label_dict
        self.max_len=max_len
        self.min_len=min_len
    def __call__(self, data):
        data=self.sorted_(data)
        text_,label=zip(*data)
        text=[]
        for w in text_:
            w=MyProcessor.date_prepare(w)
            w=MyProcessor.word_cut(w)
            text.append(w)

        seq_len=max(min(self.max_len,len(text[0])),self.min_len)
        batch_size=len(text)
        text=[[self.vocab_dict.get(j,self.nuk_id) for j in w[:seq_len]] for w in text]

        sent_tensor=torch.ones(batch_size,seq_len).long()*self.pad_id
        for i in range(batch_size):
            tensor=torch.tensor(text[i]).long()
            sent_tensor[i,:len(text[i])]=tensor
        label=[self.label_dict[w] for w in label]
        label_tensor=torch.tensor(label).long()
        return sent_tensor,label_tensor

    def sorted_(self,data):
        indicator=sorted(range(len(data)),key=lambda x:len(data[x][0]),reverse=True)
        data=[data[i] for i in indicator]

        return data

    def text_prepare(self,text_list):
        seq_len=max([len(w) for w in text_list])
        text = []
        for w in text_list:
            w = MyProcessor.date_prepare(w)
            w = MyProcessor.word_cut(w)
            text.append(w)

        seq_len = max(min(self.max_len, seq_len), self.min_len)
        batch_size = len(text)
        text = [[self.vocab_dict.get(j, self.nuk_id) for j in w[:seq_len]] for w in text]

        sent_tensor = torch.ones(batch_size, seq_len).long() * self.pad_id
        for i in range(batch_size):
            tensor = torch.tensor(text[i]).long()
            sent_tensor[i, :len(text[i])] = tensor
        return sent_tensor


def make_loader(collator_fn,train_file,eval_file,batch_size):
    train_loader=None
    if train_file and os.path.exists(train_file):
        train_loader=DataLoader(MyDataSet(train_file),batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=collator_fn)
    eval_loader=None
    if eval_file and os.path.exists(eval_file):
        eval_loader=DataLoader(MyDataSet(eval_file),batch_size=batch_size,shuffle=False,num_workers=4,
                               collate_fn=collator_fn)
    return train_loader,eval_loader



def args_parse():
    args=argparse.ArgumentParser(description="classify")
    args.add_argument("--train_file", default="./resource/train.csv")
    args.add_argument("--eval_file", default="./resource/test.csv")
    args.add_argument("--batch_size",default=64)
    args.add_argument("--vocab_path",default="./resource/vocab.txt")
    args.add_argument("--label_path", default=os.path.join(Config.path,"sentence_classify/resource/label_vocab.txt"))
    args.add_argument("--max_len", default=200)
    args.add_argument("--min_len",default=10)
    args.add_argument("--embed_dim", default=64)
    args.add_argument("--kernels", default=[3,4,5])
    args.add_argument("--out_channels", default=64)
    args.add_argument("--hidden_dim", default=64)
    args.add_argument("--drop_rate", default=0.3)
    args.add_argument("--lr", default=0.001)
    args.add_argument("--device",default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    args.add_argument("--save_path",default="./resource")
    args.add_argument("--max_epoch",default=30)
    args.add_argument("--log_print",default=50)
    args.add_argument("--step_pring",default=100)
    args.add_argument("--log", default="./resource/log")
    return args.parse_args()



def main():

    args=args_parse()
    process=MyProcessor()
    vocab=process.make_vocab(args.vocab_path)
    label_vocab=process.get_class_vocab(args.label_path)
    collator=MyCollator(vocab_dict=vocab,label_dict=label_vocab,max_len=args.max_len,min_len=args.min_len)
    train_loader,eval_loader=make_loader(collator_fn=collator,train_file=args.train_file,eval_file=args.eval_file,
                                         batch_size=args.batch_size)
    model=CNN_Model(vocab_size=len(vocab),embed_dim=args.embed_dim,kernels=args.kernels,out_channels=args.out_channels,
                    hidden_dim=args.hidden_dim,out_dim=len(label_vocab),drop_tate=args.drop_rate)

    model.to(args.device)
    print(model)
    shutil.rmtree(args.log)
    os.mkdir(args.log)

    train(model,args,train_loader,eval_loader,model_name="cls")

def train(model,args,train_loader,eval_loader,model_name):
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    writer=SummaryWriter(args.log)
    step=0
    best_loss=1e3
    logging.info("train batch {}".format(len(train_loader)))
    for epoch in range(args.max_epoch):
        model.train()
        for id,batch in enumerate(train_loader):
            text,label=batch
            text.to(args.device),label.to(args.device)
            output=model(text)
            loss=criterion(output.view(-1,output.size(-1)),label.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step+=1
            if id%args.log_print==0:
                correct=torch.max(output,dim=-1)[1].cpu().data
                acc=100*(correct==label.cpu().data).sum()/output.size(0)
                logging.info("{} train epoch {} id {} loss {} acc {}".format(model_name,epoch,id,loss.item(),acc.item()))
                writer.add_scalar("{}_train_loss".format(model_name),loss.item(),step)
                writer.add_scalar("{}_train_acc".format(model_name),acc.item(),step)

                if best_loss>loss.item():
                    best_loss=loss.item()
                    save_path=os.path.join(args.save_path,"{}_model.pt".format(model_name))
                    torch.save(model.state_dict(),save_path)
                    logging.info(save_path)


            if step%args.step_pring==0:
                acc_,loss_=valid(model,eval_loader,args)
                logging.info("{} eval loss {} acc {}".format(model_name,loss_,acc_))
                writer.add_scalar("{}_eval_loss".format(model_name), loss_, step)
                writer.add_scalar("{}_eval_acc".format(model_name), acc_, step)

def valid(model,eval_loader,args):
    model.eval()
    with torch.no_grad():
        criterion=nn.CrossEntropyLoss()
        total_batch=0
        total_acc=0
        total_loss=0

        for id,batch in enumerate(eval_loader):
            text,label=batch
            text.to(args.device),label.to(args.device)
            output=model(text)
            loss=criterion(output.view(-1,output.size(-1)),label.reshape(-1))
            total_loss+=loss.item()

            correct=(torch.max(output,dim=-1)[1].cpu().data==label.cpu().data).sum()
            total_acc+=correct.item()
            total_batch+=label.size(0)
        return 100*total_acc/total_batch,total_loss/id



if __name__ == '__main__':
    main()


