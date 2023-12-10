#coding=utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence

class LstmClassify(nn.Module):
    def __init__(self,args):
        super(LstmClassify,self).__init__()
        self.args=args
        self.embedding=nn.Embedding(self.args.vocab_size,self.args.embed_dim)
        self.lstm=nn.LSTM(input_size=self.args.embed_dim,hidden_size=self.args.hidden_dim,
                          num_layers=args.num_layers,batch_first=True,dropout=self.args.drop_out_rate)

        self.lin1=nn.Linear(self.args.hidden_dim*self.args.max_len,self.args.hidden_dim)
        self.lin2=nn.Linear(self.args.hidden_dim,self.args.label_nums)

    def forward(self,inputs,hidden,length):
        embed=self.embedding(inputs)
        packed = pack_padded_sequence(embed, lengths=length, batch_first=True)
        output, _ = self.lstm(packed,hidden)
        paded, _ = pad_packed_sequence(output, batch_first=True)
        output=paded.reshape(embed.size(0),-1)
        output=self.lin1(output)
        scores=self.lin2(output)
        return scores,hidden

    def hidden_init(self,batch_size):
        hidden=(torch.zeros(self.args.num_layers,batch_size,self.args.hidden_dim),
                torch.zeros(self.args.num_layers,batch_size,self.args.hidden_dim))

        return hidden























