
from torchtext.legacy import data
import re
import torch
import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s-%(message)s")
import json
import pandas as pd
def get_datasets():
    path="G:\software_py\pytorch\孪生神经网络\datasets/oppp相似度数据集.json"
    path1="G:\software_py\pytorch\孪生神经网络\datasets/train.tsv"
    path2 = "G:\software_py\pytorch\孪生神经网络\datasets/dev.tsv"
    path3 = "G:\software_py\pytorch\孪生神经网络\datasets/test.tsv"
    with open(path,"r",encoding="utf-8") as f:
        data=json.load(f)
        test=data["test"]
        train=data["train"]
        dev=data["dev"]
        train_dfs=pd.DataFrame(train)
        devdfs=pd.DataFrame(dev)
        testdfs=pd.DataFrame(test)
        train_dfs.to_csv(path1,sep="\t")
        devdfs.to_csv(path2,"\t")
        testdfs.to_csv(path3,sep="\t")
    return

def word_cut(text):
    return [word for word in list(str(text)) if word.strip() and word!=" "]

def build_vocab(path,num,save_path="G:\software_py\pytorch\孪生神经网络/vocab/vocab.txt"):
    vocab_dict=dict()
    dfs=pd.read_csv(path,sep="\t")
    dfs=dfs.drop(columns=["Unnamed: 0"])
    for i in range(len(dfs)):
        q1=dfs.loc[i,"q1"]
        q2=dfs.loc[i,"q2"]
        tokens=word_cut(q1)
        tokens1=word_cut(q2)
        for w in tokens:
            vocab_dict[w]=vocab_dict.get(w,0)+1
        for w in tokens1:
            vocab_dict[w] = vocab_dict.get(w, 0) + 1
    sorted_vocab=sorted(vocab_dict.items(),key=lambda x:x[1],reverse=True)
    f=open(save_path,"w",encoding="utf-8")
    for i in range(min(num,len(sorted_vocab))):
        f.write(sorted_vocab[i][0]+"\n")
    f.write("UNK"+"\n")
    f.write("PAD"+"\n")
    f.close()



def get_data(path):
    dfs=pd.read_csv(path,sep="\t")
    dfs=dfs.drop(columns=["Unnamed: 0"])
    datasets=[]
    for w in range(len(dfs)):
        q1=word_cut(dfs.loc[w,"q1"])
        q2=word_cut(dfs.loc[w,"q2"])
        label=dfs.loc[w,"label"]
        datasets.append([q1,q2,label])
    return datasets


def read_datasets(seq_len):
    path1 = "G:\software_py\pytorch\孪生神经网络\datasets/train.tsv"
    path2 = "G:\software_py\pytorch\孪生神经网络\datasets/dev.tsv"
    train=get_data(path1)
    valid=get_data(path2)
    save_path="G:\software_py\pytorch\孪生神经网络/vocab/vocab.txt"
    build_vocab(path1, 5000,save_path)
    vocab_dict=dict()
    id=0
    with open(save_path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            vocab_dict[line]=id
            id+=1
    '''填充'''
    train=padding(train,vocab_dict,seq_len)
    valid=padding(valid,vocab_dict,seq_len)
    return train, valid,vocab_dict


def padding(train,vocab_dict,seq_len):
    train_datasets = []
    for w in train:
        mid = [vocab_dict[w] if w in vocab_dict else vocab_dict["UNK"] for w in w[0]]
        mid1 = [vocab_dict[w] if w in vocab_dict else vocab_dict["UNK"] for w in w[1]]
        if len(mid) < seq_len:
            mid.extend([vocab_dict["PAD"] for _ in range(seq_len - len(mid))])
        else:
            mid = mid[:seq_len]

        if len(mid1) < seq_len:
            mid1.extend([vocab_dict["PAD"] for _ in range(seq_len - len(mid1))])
        else:
            mid1 = mid1[:seq_len]
        train_datasets.append([mid, mid1, w[2]])
    return train_datasets


class DatasetIter(object):
    def __init__(self,batches,batch_size,device):
        self.batch_size=batch_size
        self.batches=batches
        self.n_batches=len(batches)//batch_size
        self.residue=False
        if len(batches)%self.n_batches!=0:
            self.residue=True
        self.index=0
        self.device=device

    def _to_tensor(self,data):
        x1=torch.LongTensor([w[0] for w in data]).to(self.device)
        x2=torch.LongTensor([w[1] for w in data]).to(self.device)
        y=torch.LongTensor([w[2] for w in data]).to(self.device)
        return x1,x2,y


    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches+1
        else:
            return self.n_batches














if __name__ == '__main__':
    path1 = "G:\software_py\pytorch\孪生神经网络\datasets/train.tsv"
    path2 = "G:\software_py\pytorch\孪生神经网络\datasets/dev.tsv"
    path3 = "G:\software_py\pytorch\孪生神经网络\datasets/test.tsv"
    read_datasets()









