#coding=utf-8

'''定义孪生神经网络'''
import torch.nn as nn
import torch

class SiameseNetSemtanic(nn.Module):
    def __init__(self,args,weight):
        super(SiameseNetSemtanic,self).__init__()
        vocab_size=args.vocab_size
        embedding_size=args.embeding_size
        dropout=args.dropout
        self.hidden_size=args.hidden_size
        self.num_layers=args.num_layers
        sentence_size=args.output_size
        seq_len=args.seq_len
        self.embed=nn.Embedding(vocab_size,embedding_size)
        #加载词嵌入，使模型在训练的时候参数不进行微调
        #self.embed=nn.Embedding.from_pretrained(weight)
        #self.embed.weight.requires_grad=False

        self.dropOut=nn.Dropout(dropout)
        self.lstm=nn.LSTM(embedding_size,self.hidden_size,self.num_layers,dropout=dropout,
                          batch_first=True)

        self.linear=nn.Linear(self.hidden_size*seq_len,sentence_size)

    def forward(self,input1,input2,hidden1,hidden2):
        out1,hidden1=self.encode(input1,hidden1)
        out2,hidden2=self.encode(input2,hidden2)
        # #相似度计算
        # #余弦分子
        # self.distance=torch.sqrt(torch.sum(torch.square(out1-out2),1,
        #                                    keepdim=True))
        # #分母
        # self.distance=torch.div(self.distance,torch.add(torch.sqrt(torch.sum(
        #                           torch.square(out1),dim=1,keepdim=True)),
        #                         torch.sqrt(torch.sum(torch.square(out2),dim=1,keepdim=True))))
        # #改变维度
        # self.distance=self.distance.reshape([-1])
        # return self.distance,hidden1,hidden2



    def encode(self,x,hidden):
        seq_len=x.size(1)
        embed=self.dropOut(self.embed(x))
        output,hidden=self.lstm(embed,hidden)
        output=output.reshape(-1,seq_len*self.hidden_size)
        output=self.dropOut(output)
        output=self.linear(output)
        return output,hidden

    def init_hidden(self,batch_size):
        return (torch.zeros(self.num_layers,batch_size,self.hidden_size),
                torch.zeros(self.num_layers,batch_size,self.hidden_size))



if __name__ == '__main__':
    from model.args import Args
    args=Args()
    model=SiameseNetSemtanic(args,weight="")
    print(model)
    param=model.state_dict()
    for i,j in enumerate(param):
        print(i,j)

    # 均值损失函数，自定义实现


