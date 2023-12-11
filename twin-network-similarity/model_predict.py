

from model.sim_model import SimModel,Config
from model.model_train import load_model
import torch
import torch.nn as nn
from model.data_prepare import word_cut

def model_predict(path,voacb_path):
    args=Config()
    vocab_dict=dict()
    id=0
    with open(voacb_path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            vocab_dict[line]=id
            id+=1
    args.vocab_size=len(vocab_dict)
    args.batch_size=1
    model=SimModel(args).to(args.device)
    map_location=lambda storage,loc:storage
    model=load_model(model,path,map_location)
    if model!=None:
        model.eval()
        def wrapper(text1,text2):
            with torch.no_grad():
                text1=to_tensor(text1,args.seq_len,vocab_dict)
                text1=torch.LongTensor([text1]).to(args.device)
                text2=to_tensor(text2,args.seq_len,vocab_dict)
                text2=torch.LongTensor([text2]).to(args.device)
                logit=model(text1,text2)   #相似度
                pred=1 if logit.item()>args.margin else 0
                print(pred)
                return logit.item()
        return wrapper

def to_tensor(text,seq_len,vocab_dict):
    text=word_cut(text)
    mid=[vocab_dict[w] if w in vocab_dict else vocab_dict["UNK"] for w in text]
    if len(mid) < seq_len:
        mid.extend([vocab_dict["PAD"] for _ in range(seq_len - len(mid))])
    else:
        mid = mid[:seq_len]
    return mid

if __name__ == '__main__':
    x1="唱一首歌曲"
    x2="歌曲关掉"
    model_path= "G:\software_py\pytorch\孪生神经网络/modelout/model/beststep100.bt"
    voacb_path= "G:\software_py\pytorch\孪生神经网络/vocab/vocab.txt"
    sim=model=model_predict(model_path,voacb_path)(x1,x2)
    print(sim)







