


import torch.nn as nn
from TorchCRF import CRF
class BiLSTMCRF(nn.Module):
    def __init__(self,args):
        super(BiLSTMCRF,self).__init__()
        self.embedding=nn.Embedding(num_embeddings=args.vocab_size,embedding_dim=args.embeding_dim)
        self.bilstm=nn.LSTM(input_size=args.embeding_dim,hidden_size=args.hidden_size,batch_first=True,
                            num_layers=args.num_layers,bidirectional=True)
        self.linear=nn.Linear(in_features=2*args.hidden_size,out_features=args.label_nums)
        self.crf=CRF(args.label_nums)
    def forward(self,input_ids,attention_mask,labels=None):
        embed_out=self.embedding(input_ids)
        seq_out,hidden_out=self.bilstm(embed_out)   #[batch,seq_len,hidden]
        out=self.linear(seq_out)
        if labels!=None:
            loss=-1*self.crf(out,labels,mask=attention_mask.byte()).mean()
            return (loss,)
        else:
            logits=self.crf.viterbi_decode(out,mask=attention_mask.byte())
            return (logits,)














