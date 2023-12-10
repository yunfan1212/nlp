#coding=utf8

'''transe知识推理，torch 实现'''
import json
import datasets
from transformers import HfArgumentParser
from dataclasses import dataclass,field
from typing import Optional
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import copy
import numpy as np
from tensorboardX import SummaryWriter
import torch
import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(filename)s-%(message)s")
from knowledge_reason.kge.transe import TransE
import os
from tqdm import tqdm


@dataclass
class DataArgument():
    trainfile:Optional[str]=field(default="")
    devfile:Optional[str]=field(default="")
    createNewDict:Optional[bool]=field(default=False)
    entity_dict_path:Optional[str]=field(default="./dict/entity2id.json")
    rel_dict_path:Optional[str]=field(default="./dict/rel2id.json")
    lr: Optional[float] = field(default=0.001)
    logPath: Optional[str] = field(default="./log")
    epoches: Optional[int] = field(default=10)
    batch_size: Optional[int] = field(default=128)
    steplog: Optional[int] = field(default=100)
    model_path: Optional[str] = field(default="./model")

'''创建词典'''
def generateDict(triples,valid_triples,test_triples,entity_dict_path,rel_dict_path):
    head=triples["head"]+valid_triples["head"]+test_triples["head"]
    relation=triples["relation"]+valid_triples["relation"]+test_triples["relation"]
    tail=triples["tail"]+valid_triples["tail"]+test_triples["tail"]

    entity=list(set(head+tail))
    relation=list(set(relation))

    entity=[w for w in entity if type(w)==str]
    relation=[w for w in relation if type(w)==str]

    entity=sorted(entity)
    relation=sorted(relation)
    entity_dict={v:k  for k,v in enumerate(entity)}
    rel_dict={v:k  for k,v in enumerate(relation)}
    json.dump(entity_dict, open(entity_dict_path, "w"))
    json.dump(rel_dict, open(rel_dict_path, "w"))


'''创建数据'''
class MyDataset(Dataset):
    def __init__(self,raw_datasets,entity2id,rel2id):
        #super(MyDataset,self).__init__()
        self.entity2id=entity2id
        self.rel2id=rel2id

        self.posSample=pd.DataFrame(raw_datasets)        #正样本
        #去除空格
        self.posSample.dropna(axis=0, how="any", inplace=True)
        self.posSample=self.posSample.applymap(lambda x:x.strip())
        self.posSample.replace({"":np.nan},inplace=True)
        self.posSample.dropna(axis=0,how="any",inplace=True)


        self.transformToIndex(self.posSample,{"head":self.entity2id,
                                        "relation":self.rel2id,
                                         "tail":self.entity2id})

        self.datasets=[torch.tensor(self.posSample.loc[i,:]) for i in range(len(self.posSample))]


    def __len__(self):
        return len(self.posSample)

    def __getitem__(self, item):

        return self.datasets[item][0],self.datasets[item][1],self.datasets[item][2]


    @staticmethod      #将数据转为id
    def transformToIndex(pdfData:pd.DataFrame,replaceDict:dict):
        for col in replaceDict.keys():
            pdfData[col]=pdfData[col].apply(lambda x:replaceDict[col][x])


def main():
    parser=HfArgumentParser((DataArgument))
    data_args=parser.parse_args_into_dataclasses()[0]
    print(data_args)

    #加载数据集
    # raw_datasets=datasets.load_dataset("VLyb/FB15k")
    # raw_datasets.save_to_disk("FB15K")
    # print(raw_datasets)
    raw_datasets=datasets.load_from_disk("FB15K")
    print(raw_datasets)
    if data_args.createNewDict:
        generateDict(raw_datasets["train"],raw_datasets["validation"],raw_datasets["test"],data_args.entity_dict_path,data_args.rel_dict_path)

    entity2id = json.load(open(data_args.entity_dict_path, "r"))
    rel2id = json.load(open(data_args.rel_dict_path, "r"))

    trainsets=raw_datasets["train"]#.select(list(range(100)))
    trainsets=MyDataset(trainsets,entity2id,rel2id)
    trainloader=DataLoader(trainsets,batch_size=data_args.batch_size,shuffle=True,num_workers=10,drop_last=True)

    testsets=raw_datasets["test"].select(list(range(100)))
    testsets = MyDataset(testsets, entity2id, rel2id)
    testloader = DataLoader(testsets, batch_size=data_args.batch_size, shuffle=True, num_workers=10)

    ###############加载模型
    model=TransE(num_nodes=len(entity2id),num_relations=len(rel2id),hidden_channels=128)
    model.cuda()
    print(model)
    model.is_parallelizable = True  # 支持并行
    model.model_parallel = True
    #开始训练
    train(trainloader,model,args=data_args)

    #开始测试
    res=test(testloader,model,args=data_args)
    logging.info(f" mean_rank {res[0]}   mrr {res[1]}  tok@10:{res[2]}")

def train(trainloader,model,args):
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    writer=SummaryWriter(args.logPath)
    step=0
    best_loss=999
    for epoch in range(args.epoches):
        model.train()
        for id,batch in tqdm(enumerate(trainloader),desc="training.."):
            heads,relations,tails=batch
            heads=heads.cuda()
            relations=relations.cuda()
            tails=tails.cuda()
            loss=model.loss(heads,relations,tails)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step+=1
            if step%args.steplog==0:
                logging.info(f"train epoch {epoch}  batch {id} loss {loss.item()}")
                writer.add_scalar("train_loss",loss.item(),step)
                if loss.item()<best_loss:
                    best_loss=loss.item()
                    path=os.path.join(args.model_path,"model_torch.pt")
                    torch.save(model.state_dict(),path)
    return


def test(testloader,model,args):
    mean_ranks,mrrs,hits_k=[],[],[]
    for id, batch in tqdm(enumerate(testloader), desc="test.."):
        heads, relations, tails = batch
        heads = heads.cuda()
        relations = relations.cuda()
        tails = tails.cuda()
        mean_rank, mrr, hits_at_k = model.test(heads, relations, tails,args.batch_size)
        mean_ranks.append(mean_rank)
        mrrs.append(mrr)
        hits_k.append(hits_at_k)
    return np.mean(mean_ranks),np.mean(mrrs),np.mean(hits_k)

if __name__ == '__main__':
    main()

#   mean_rank 378.2799987792969   mrr 0.20935939252376556  tok@10:0.38












