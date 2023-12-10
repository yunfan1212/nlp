

from knowledge_reason.kbc.kbc_model import ComplEx,N3
from transformers import HfArgumentParser
from dataclasses import dataclass,field
import datasets
from typing import Optional,List
import json
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch import nn
import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(filename)s-%(message)s")
from tqdm import tqdm
import os

@dataclass
class DataArguments:
    dataset:Optional[str]=field(default="FB15K")
    regularizer:Optional[str]=field(default="N3")
    max_epoches:Optional[int]=field(default=100)
    valid:Optional[float]=field(default=5)
    rank:Optional[int]=field(default=500)
    batch_size:Optional[int]=field(default=1000)
    reg:Optional[float]=field(default=1e-2)
    init:Optional[float]=field(default=1e-1)
    learning_rate:Optional[float]=field(default=1e-1)
    decay1:Optional[float]=field(default=0.9)
    decay2:Optional[float]=field(default=0.999)
    createNewDict:bool=field(default=False)
    entity_dict_path: Optional[str] = field(default="./dict/entity2id.json")
    rel_dict_path: Optional[str] = field(default="./dict/rel2id.json")
    log_path:Optional[str]=field(default="./log")
    steplog: Optional[int] = field(default=100)
    model_path:Optional[str]=field(default="./model")

'''创建词典,包含训练集、测试集的所有关系，节点'''
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
    def __init__(self,raw_datasets,entity2id,rel2id,types):
        #super(MyDataset,self).__init__()
        self.entity2id=entity2id
        self.rel2id=rel2id
        self.type=types

        self.posSample=pd.DataFrame(raw_datasets)        #正样本
        #去除空格
        self.posSample.dropna(axis=0, how="any", inplace=True)
        self.posSample=self.posSample.applymap(lambda x:x.strip())
        self.posSample.replace({"":np.nan},inplace=True)
        self.posSample.dropna(axis=0,how="any",inplace=True)


        self.transformToIndex(self.posSample,{"head":self.entity2id,
                                        "relation":self.rel2id,
                                         "tail":self.entity2id})

        nums=len(self.rel2id)
        self.n_predicates=nums*2

        if self.type=="train":
            self.posSample=self.get_train()
        else:
            #没有反向
            pos=[[self.posSample.loc[i,"head"],self.posSample.loc[i,"relation"],self.posSample.loc[i,"tail"]] for i in range(len(self.posSample))]
            pos=[w for w in pos if type(w[0])==np.int64 and type(w[0])==np.int64 and type(w[0])==np.int64]
            self.posSample=np.array(pos)

        self.posSample = torch.from_numpy(self.posSample.astype("int64"))

    def __len__(self):
        return len(self.posSample)

    def __getitem__(self, item):
        return self.posSample[item]

    @staticmethod      #将数据转为id
    def transformToIndex(pdfData:pd.DataFrame,replaceDict:dict):
        for col in replaceDict.keys():
            pdfData[col]=pdfData[col].apply(lambda x:replaceDict[col][x])

    def get_train(self):
        self.posSample_train=np.array([[self.posSample.loc[i,"head"],self.posSample.loc[i,"relation"],self.posSample.loc[i,"tail"]] for i in range(len(self.posSample))])
        copy=np.copy(self.posSample_train)
        tmp = np.copy(copy[:, 0])  # 头节点
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two. 反向 指向
        return np.vstack((self.posSample_train, copy))       #正样本


'''创建目标节点集合,用于测试'''
def get_targets_sets(raw_datasets,entity2id,rel2id):

    n_relations=len(rel2id)
    #lhs 右指向， rhs 左指向
    to_skip={"lhs":defaultdict(set),"rhs":defaultdict(set)}
    files=["train","validation","test"]
    for f in tqdm(files,desc="data prepare"):
        heads = raw_datasets[f]["head"]
        relations = raw_datasets[f]["relation"]
        tails = raw_datasets[f]["tail"]
        for index in range(len(heads)):
            if heads[index] not in entity2id or tails[index] not in entity2id or relations[index] not in rel2id:
                continue
            lhs=entity2id[heads[index]]
            rel=rel2id[relations[index]]
            rhs=entity2id[tails[index]]
            #正向，目标节点集合
            to_skip["rhs"][(lhs,rel)].add(rhs)
            #反向，目标节点集合
            to_skip["lhs"][(rhs,rel+n_relations)].add(lhs)
    #对目标节点排序
    to_skip_final={'lhs': {}, 'rhs': {}}
    for kk,skip in tqdm(to_skip.items(),desc="data prepare"):
        for k,v in skip.items():
            to_skip_final[kk][k]=sorted(list(v))   #保持每次输出一致
    return to_skip_final


def main():
    parsers=HfArgumentParser((DataArguments))
    data_args=parsers.parse_args_into_dataclasses()[0]
    print(data_args)
    #加载数据集
    raw_datasets = datasets.load_from_disk("FB15K")

    if data_args.createNewDict:
        generateDict(raw_datasets["train"], raw_datasets["validation"], raw_datasets["test"],
                     data_args.entity_dict_path, data_args.rel_dict_path)

    entity2id = json.load(open(data_args.entity_dict_path, "r"))
    rel2id = json.load(open(data_args.rel_dict_path, "r"))

    #目标值的集合
    to_skip=get_targets_sets(raw_datasets,entity2id,rel2id)

    trainsets = raw_datasets["train"].select(list(range(100)))
    trainsets = MyDataset(trainsets, entity2id, rel2id,"train")
    trainloader = DataLoader(trainsets, batch_size=data_args.batch_size, shuffle=True, num_workers=10, drop_last=True)

    testsets = raw_datasets["test"].select(list(range(10000)))
    testsets = MyDataset(testsets, entity2id, rel2id,"test")
    testloader = DataLoader(testsets, batch_size=data_args.batch_size, shuffle=False, num_workers=4)


    #加载模型
    data_args.n_predicates = len(rel2id)*2
    model=ComplEx((len(entity2id),data_args.n_predicates,len(entity2id)),rank=data_args.rank,init_size=data_args.init)
    model.float()

    regularizer=N3(data_args.reg)
    model.is_parallelizable = True  # 支持并行
    model.model_parallel = True

    #开始训练

    train(trainloader,testloader,data_args,to_skip,model=model,regularizer=regularizer)

def train(trainLoder,testLoader,args,to_skip,model,regularizer):
    writer=SummaryWriter(args.log_path)
    optimizer=torch.optim.Adagrad(model.parameters(),lr=args.learning_rate)
    cretrion=nn.CrossEntropyLoss(reduction="mean")
    best_loss=1000
    step=0
    for epoch in range(args.max_epoches):
        for batch in tqdm(trainLoder,desc="training.."):
            batch=batch.cuda()
            truth=batch[:,2].cuda()
            predictions, factors = model.forward(batch)

            l_fit=cretrion(predictions,truth)
            l_reg=regularizer(factors)
            loss=l_fit+l_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step+=1
            if step%args.steplog==0:
                logging.info(f"train epoch {epoch}  batch {step} loss {loss.item()}")
                writer.add_scalar("train_loss", loss.item(), step)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    path = os.path.join(args.model_path, "model_torch_compex.pt")
                    torch.save(model.state_dict(), path)
        #测试
        if (epoch+1)%args.valid==0:
            test(testLoader,model,to_skip,args)

def test(testLoader,model,to_skip,args):

    missing=["rhs","lhs"] #对应to_skip
    mrrs=torch.tensor(0)
    hits=torch.tensor([0,0,0])
    step=0
    for m in tqdm(missing,desc="test"):
        for batch in testLoader:
            q=batch.cuda()
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += args.n_predicates//2     #左右节点交换

            ranks = model.get_ranking(q, to_skip[m], batch_size=args.batch_size)  # 获取前K 个数据，
            mean_reciprocal_rank = torch.mean(1. / ranks).item()     #排序倒数
            at= (1, 3, 10)
            hits_at = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

            hits=hits_at+hits
            mrrs=mrrs+mean_reciprocal_rank
            step+=1
        m1 = mrrs/step
        h1 = hits/step
        logging.info(f"{m}  MRR:{m1.item()},hits@[1,3,10]:{h1.cpu().tolist()}")

    return



if __name__ == '__main__':
    main()





















