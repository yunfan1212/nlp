
import torch
import torch.nn as nn

class LstmModel(nn.Module):
    def __init__(self,args):
        super(LstmModel,self).__init__()
        self.args=args

        self.embedding=nn.Embedding(self.args.vocab_size,self.args.embed_dim)
        self.lstm=nn.LSTM(self.args.embed_dim,hidden_size=self.args.hidden_dim,num_layers=self.args.num_layers,
                          batch_first=True,dropout=self.args.dropout_rate)

        self.lin=nn.Linear(self.args.max_len*self.args.hidden_dim,self.args.hidden_dim)
        self.lin1=nn.Linear(self.args.hidden_dim,self.args.label_nums)

    def forward(self,inputs,hidden):
        embed=self.embedding(inputs)
        output,hidden=self.lstm(embed,hidden)
        output=output.reshape(inputs.size(0),-1)
        output=self.lin(output)
        output=self.lin1(output)
        return output,hidden

    def hidden_init(self,batch_size):
        return (torch.zeros(self.args.num_layers,batch_size,self.args.hidden_dim),
                torch.zeros(self.args.num_layers,batch_size,self.args.hidden_dim))















