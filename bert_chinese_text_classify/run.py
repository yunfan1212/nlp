# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default="bert_CNN", help='choose a model: Bert, ERNIE')
args = parser.parse_args()


#采用什么网络，model输入网络对应的文件名, 如bert，bert_CNN,bert_RNN,bert_RCNN
if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)      #动态导入对应的模块

    config = x.Config(dataset)                   #加载对应模块的config

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    #加载自定义数据集
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train

    model = x.Model(config).to(config.device)        #加载对应的模型

    train(config, model, train_iter, dev_iter, test_iter)
