#coding=utf-8

from model.data_prepare import read_datasets
from model.data_prepare import DatasetIter
from model.sim_model import SimModel
import torch
import numpy as np
from model.sim_model import Config
from model.model_train import train

def main():
    args=Config()
    trainDatasets,validDatasets,vocab_dict=read_datasets(args.seq_len)
    trainIter=DatasetIter(trainDatasets,args.batch_size,args.device)
    validIter=DatasetIter(validDatasets,args.batch_size,args.device)
    args.vocab_size=len(vocab_dict)
    model=SimModel(args).to(args.device)
    train(model,trainIter,validIter,args)
if __name__ == '__main__':
    main()









if __name__ == '__main__':
    main()


