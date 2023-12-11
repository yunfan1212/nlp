import torch
import torch.nn as nn
import torch.nn.functional as F

class Config():
    vocab_size=5000
    embedding_size=128
    keep_prob=0.8
    out_channels=512
    filter_size=[2,3,4]
    output_size=128
    learning_rate=0.001
    batch_size=200
    epoches=50
    seq_len=30
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_interval=10
    test_interval=50
    log_path="./modelout/log"
    save_dir = "./modelout/model"
    margin=0.5          #小于0.3 则赋值为0，大于0.3 则赋值为1
    class_nun=2

class SimModel(nn.Module):
    def __init__(self,args):
        super(SimModel,self).__init__()

        self.embedding=nn.Embedding(args.vocab_size,args.embedding_size)
        self.dropout=nn.Dropout(args.keep_prob)
        self.convd=nn.ModuleList([nn.Conv1d(in_channels=args.embedding_size,
                                            out_channels=args.out_channels,kernel_size=size) for size in args.filter_size])
        self.linear=nn.Linear(len(args.filter_size)*args.out_channels,args.output_size)
        self.linear2=nn.Linear(2*args.output_size,args.output_size)
        self.linear3=nn.Linear(args.output_size,args.class_nun)


    def forward(self,text1,text2):
        embedding_1=self.encoder(text1)     #(batch,hidden_dim)
        embedding_2=self.encoder(text2)     #(batch,hidden_dim)
        #cos=torch.cosine_similarity(embedding_1,embedding_2)     #[batch] --> [0.3.0.2,...]
        out=torch.cat([embedding_1,embedding_2],dim=1)
        out=F.relu(self.linear2(out))
        out=self.dropout(out)
        out=self.linear3(out)
        return out

    def encoder(self,text):
        output = self.dropout(self.embedding(text))
        output = output.permute(0, 2, 1)  # batch,embed,seq_len
        output = [torch.max(conv(output), dim=2)[0] for conv in self.convd]
        output = torch.cat(output, dim=1)
        output = self.linear(output)
        return output


class ConstrastiveLoss(nn.Module):
    def __len__(self):
        super(ConstrastiveLoss,self).__init__()
    def forward(self,Ew,y):
        l_1=0.25*(1-Ew)*(1-Ew)
        l_0=torch.where(Ew<Config.margin*torch.ones_like(Ew),torch.full_like(Ew,0),Ew)*torch.where(
            Ew<Config.margin*torch.ones_like(Ew),torch.full_like(Ew,0),Ew)
        loss=y*1.0*l_1+(1-y)*1.0*l_0
        return loss.sum()





