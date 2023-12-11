from gensim.models import word2vec
import gensim.models as models
import jieba
import gensim
import numpy as np
import tensorflow as tf
import os
path=os.path.abspath(os.path.dirname(__file__))
path=os.path.join(path,"word2vec/tian2.txt")

sentences = word2vec.Text8Corpus(path)
model = word2vec.Word2Vec(sentences,  min_count=1,size=10)        #size为词频，创建词典
print(model.wv["的"])
model.save("word2vec/embedding.txt")  # 保存模型
# 保存词向量
model.wv.save_word2vec_format("word2vec/word2vec.bin",binary=False)      #保存的是模型的值
model.wv.save_word2vec_format("word2vec/word2vec1.txt",binary=False)   #二进制保存
print("-----------")
mm=word2vec.Word2Vec.load("word2vec/embedding.txt")

####核心：保存时二进制，读取时二进制   保存时非二进制，读取时非二进制
wordvec=gensim.models.KeyedVectors.load_word2vec_format("word2vec/word2vec.bin",binary=False)

#从模型中获得词向量
def getwordEmbedding(words,embedding_size):
    vocab=[]
    wordEmbedding=[]
    #添加pad 和 unk
    vocab.append("PAD")
    vocab.append("UNK")
    wordEmbedding.append(np.zeros(embedding_size))
    wordEmbedding.append(np.random.randn(embedding_size))
    for word in words:
        try:
            vector=wordvec.wv[word]
            vocab.append(word)
            wordEmbedding.append(vector)
        except:
            print(word+"不存在于词向量中")
    return vocab,np.array(wordEmbedding)

words="随着 人工智能 的 兴起"
words=words.split(" ")
vocab,embedding=getwordEmbedding(words,10)
print(vocab)
print(embedding)
print("--------------")



