#coding=utf8
import torch
import torch.nn as nn


class CNN_Model(nn.Module):
    def __init__(self,vocab_size,embed_dim,kernels,out_channels,hidden_dim,out_dim,drop_tate):
        super(CNN_Model,self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_dim)
        self.dropout=nn.Dropout(drop_tate)
        self.cnn=nn.ModuleList([nn.Conv1d(in_channels=embed_dim,out_channels=out_channels,kernel_size=kernel_size)
                                for kernel_size in kernels])
        self.linear=nn.Linear(in_features=len(kernels)*out_channels,out_features=hidden_dim)
        self.lin_out=nn.Linear(hidden_dim,out_dim)

    def forward(self,text):
        embed=self.dropout(self.embed(text))
        input=embed.permute(0,2,1)              #(batch,out_channel,seq_len-kernel+1)
        output=[torch.max(conv(input),dim=2)[0] for conv in self.cnn]
        output=torch.cat(output,dim=-1)   #batch,len(kernels)*out_channel
        output=self.linear(output)
        out=self.lin_out(output)
        return out















