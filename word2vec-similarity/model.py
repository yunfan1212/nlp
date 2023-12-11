from gensim.models import word2vec
import gensim.models as models
import jieba
import numpy as np
import gensim
import jieba
import numpy as np
from scipy.linalg import norm
import pandas as pd
from common_data_handle import get_all_path_list
#数据处理

class DataPrapare:
    def main(self,words_path,dict_save_path,data_path,save_path1):
        self.creat_dict(words_path,dict_save_path,name="Sheet1")
        self.cteate_train_dataset(dict_save_path,data_path,save_path1)

    def cteate_train_dataset(self,dict_path,data_path,save_path):
        path_list=get_all_path_list(dict_path)
        for path in path_list:
            jieba.load_userdict(path)
        self.creat_dataset(data_path,save_path)

    def creat_dict(self,path,save_path,name="Sheet1"):
        self.read_excel(path,name,save_path)
    def read_excel(self,path,sheet_name,save_path):
        data=[]
        data1=[]
        dfs=pd.read_excel(path,sheet_name)
        dfs=list(set(list(dfs.loc[:,"敏感字段信息"].values)))
        for tokens in dfs:
            if type(tokens)==str:
                token=tokens.split("|")
                data.extend(token)
        data=list(set(data))
        for tokens in data:
            token=tokens.split("/")
            data1.extend(token)
        data1=list(set(data1))
        self.write_txt(save_path,data1)
        return data1

    def write_txt(self,path,data):
        with open(path,"w",encoding="utf-8") as f:
            for token in data:
                f.write(token+"\n")
        return
    def creat_dataset(self,path,save_path):
    #创建符合训练条件的数据集
        with open(path,"r",encoding="utf-8") as f:
            ff=open(save_path,"w",encoding="utf-8")
            for eachline in f:
                eachline=eachline.strip("\n")
                seg_list=jieba.lcut(eachline)
                ff.write(" ".join(seg_list))
            ff.close()



class Vector_model:
    def __init__(self,dict_path):
        path_list=get_all_path_list(dict_path)
        for p in path_list:
            jieba.load_userdict(p)
    def model_train(self,path):
        sentences = word2vec.Text8Corpus(path)
        model = word2vec.Word2Vec(sentences, size=10)
        model.save("./word2vec/embedding.model")  # 保存模型
        # 保存词向量  保存二进制，读取二进制
        model.wv.save_word2vec_format("./word2vec/word2vec.txt",binary=False)
        print("save done")
        return
    def model_train_again(self,path):
        #追加训练
        path_list=get_all_path_list(path)
        for path in path_list:
            sentences = word2vec.Text8Corpus(path)
            model = models.Word2Vec.load("word2vec/embedding.model")  # 加载模型
            model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # 训练
            model.save("word2vec/embedding.model")  # 保存模型
            # 保存词向量  保存二进制，读取二进制
            model.wv.save_word2vec_format("./word2vec/word2vec.txt", binary=False)
        print("save done")
        return



if __name__=="__main__":

    import os
    path=os.path.abspath(os.path.dirname(__file__))
    #数据预处理
    path1=os.path.join(path,"dataset/config.xlsx")
    save_path=os.path.join(path,"dict/jieba_dict.txt")
    path2=os.path.join(path,"dataset/tian1.txt")
    save_path1=os.path.join(path,"word2vec/trainset.txt")
    data=DataPrapare()
    data.main(path1,save_path,path2,save_path1)

    #模型训练
    dict_path=os.path.join(path,"dict")
    model=Vector_model(dict_path)
    train_path=os.path.join(path,"word2vec/trainset.txt")
    model.model_train(train_path)
    path2=os.path.join(path,"train_dataset")
    model.model_train_again(path2)