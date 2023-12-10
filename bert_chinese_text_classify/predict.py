# coding: UTF-8
import time
import torch
import numpy as np

from train_eval import predict_
from importlib import import_module
import argparse
from utils_predict import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default="bert", help='choose a model: Bert, ERNIE')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    start_time = time.time()
    print("Loading data...")
    #train_data= build_dataset(config)
    config.batch_size=1
    # train
    model = x.Model(config).to(config.device)
    predict=predict_(config,model)
    while True:
        start0=time.time()
        text=input("请输入：")
        start = time.time()
        print("输入时间：",start-start0)

        train_data = build_dataset(config)
        train_data=train_data(text)
        train_iter = build_iterator(train_data, config)
        result=predict(train_iter)
        end = time.time()
        print(result)
        print(" 网络 time:",end-start)
