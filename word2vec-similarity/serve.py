import gensim
import jieba
import numpy as np
from common_data_handle import get_all_path_list
from scipy.linalg import norm
import time


class Service:
    def __init__(self,dict_path,model_file,embedding_size=64):
        path_list=get_all_path_list(dict_path)
        for p in path_list:
            #jieba.load_userdict(p)
            pass
        Start=time.time()                                    #二进制保存，二进制读取
        self.model= gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
        end=time.time()
        print("model load time:",end-Start)
        self.embedding_size=embedding_size
    def vector_similarity(self,s1, s2):
        def sentence_vector(s):
            words = jieba.lcut(s)
            v = np.zeros(self.embedding_size)
            for word in words:
                try:
                    v += self.model[word]
                except:
                    #对于未知unk
                    v+=np.random.randn(self.embedding_size)
            v /= len(words)
            return v
        v1, v2 = sentence_vector(s1), sentence_vector(s2)
        print(v1, v2)
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

#模型应用
import os
path = os.path.abspath(os.path.dirname(__file__))
path1 = os.path.join(path, "dict")
model_file = 'word2vec/embedding_64.bin'
model=Service(path1,model_file)
while True:
    s1=input("s1")
    s2=input("s2")
    s = time.time()
    result = model.vector_similarity(s1, s2)
    ss = time.time()
    print(ss - s)
    print(result)
    print("-----------------")

