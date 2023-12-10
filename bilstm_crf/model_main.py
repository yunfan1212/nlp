from entity_ner.bilstm_crf.models import BiLSTMCRF
import torch
from torch.utils.data import DataLoader,Dataset
from tensorboardX import SummaryWriter
import argparse
import logging
import os
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(filename)s-%(levelname)s-%(message)s")
from entity_ner.bilstm_crf.evaluating import Metrics

UNK="<UNK>"
PAD="<PAD>"
SOS="<SOS>"
EOS="<EOS>"

class MyProcessor():
    def __init__(self):
        pass
    @staticmethod
    def read_files(filename):
        words_lists=[]
        tags_lists=[]
        with open(filename,"r",encoding="utf8") as f:
            word_list=[]
            tag_list=[]
            for line in f:
                line=line.strip()
                line=line.split(" ")
                if len(line)==2:
                    word_list.append(line[0])
                    tag_list.append(line[1])
                else:
                    words_lists.append(word_list)
                    tags_lists.append(tag_list)
                    word_list=[]
                    tag_list=[]
        assert len(words_lists)==len(tags_lists)
        return words_lists,tags_lists


    def get_vocab(self,vacab_path,train_file):
        if os.path.exists(vacab_path):
            vocab=self.read_text(vacab_path)
            return vocab
        else:
            words_lists,tages_lists=MyProcessor.read_files(train_file)
            tokens=sorted(list(set([j for w in words_lists for j in w])))
            tokens=[PAD,UNK,SOS,EOS]+tokens
            token_vocab={k:i for i,k in enumerate(tokens)}
            self.save_txt(vacab_path,token_vocab)
            return token_vocab




    def get_label(self,label_path,train_file):
        if os.path.exists(label_path):
            vocab = self.read_text(label_path)
            return vocab
        else:
            words_lists, tages_lists = MyProcessor.read_files(train_file)
            tokens = sorted(list(set([j for w in tages_lists for j in w])))
            tokens =[PAD,SOS,EOS]+tokens
            token_vocab = {k: i for i, k in enumerate(tokens)}
            self.save_txt(label_path, token_vocab)
            return token_vocab



    def save_txt(self,path,vocab):
        with open(path,"w",encoding="utf8") as f:
            for k,v in vocab.items():
                f.write("{}\t{}\n".format(k,v))
        return

    def read_text(self,path):
        with open(path,"r",encoding="utf8") as f:
            vocab=dict()
            for line in f:
                line=line.strip()
                line=line.split("\t")
                if len(line)==2:
                    vocab[line[0]]=int(line[1])
        return vocab




class MyDataSet(Dataset):
    def __init__(self,filename):
        self.words_lists, self.tages_lists = MyProcessor.read_files(filename)

    def __len__(self):
        return len(self.words_lists)

    def __getitem__(self, item):
        return self.words_lists[item],self.tages_lists[item]



class MyDataSets(Dataset):
    def __init__(self,tokenslist,labelslists):
        self.words_lists, self.tages_lists =tokenslist,labelslists

    def __len__(self):
        return len(self.words_lists)

    def __getitem__(self, item):
        return self.words_lists[item],self.tages_lists[item]



class MyCollator():
    def __init__(self,vocab,lebel2id):
        self.vocab=vocab
        self.label2id=lebel2id

    def __call__(self, data):
        data=self.sorted_by_length(data)
        words_list, tag_list = list(zip(*data))
        words_id_list, attention_masks, tag_id_list=self.convert_examples_to_features(words_list,tag_list)
        input_ids=torch.tensor(words_id_list).long()
        attention_masks=torch.tensor(attention_masks).long()
        labels=torch.tensor(tag_id_list).long()
        return [input_ids,attention_masks,labels]



    def convert_examples_to_features(self,words_list,tag_list):
        assert len(words_list)==len(tag_list)
        max_len=len(words_list[0])+2
        words_id_list=[]
        tag_id_list=[]
        attention_masks=[]
        for i in range(len(words_list)):
            _words=words_list[i]
            _tags=tag_list[i]
            _words=[self.vocab.get(j,self.vocab[UNK]) for j in _words]
            _tags=[self.label2id[w] for w in _tags]
            _words=[self.vocab[SOS]]+_words+[self.vocab[EOS]]
            _tags=[self.label2id[SOS]]+_tags+[self.label2id[EOS]]
            attention_mask = [1 for i in range(len(_words))]
            attention_mask_=[]
            if len(_words)<max_len:
                num=max_len-len(_words)
                attention_mask_ = [0 for i in range(num)]
                _words=_words+[self.vocab[PAD]]*num
                _tags=_tags+[self.label2id[PAD]]*num

            words_id_list.append(_words)
            tag_id_list.append(_tags)
            attention_mask.extend(attention_mask_)
            attention_masks.append(attention_mask)
        return words_id_list,attention_masks,tag_id_list


    def sorted_by_length(self,data):
        indices=sorted(range(len(data)),key=lambda x:len(data[x][0]),reverse=True)
        pairs=[data[i] for i in indices]
        return pairs

def make_loader(collator,train_file,valid_file,test_file,batch_size):
    train_loader=DataLoader(MyDataSet(train_file),batch_size=batch_size,shuffle=True,num_workers=4,
                            collate_fn=collator)
    valid_loader=DataLoader(MyDataSet(valid_file),batch_size=batch_size,shuffle=False,num_workers=4,
                            collate_fn=collator)
    test_loader = DataLoader(MyDataSet(test_file), batch_size=batch_size, shuffle=False, num_workers=4,
                              collate_fn=collator)
    return train_loader,valid_loader,test_loader


def get_args():
    args=argparse.ArgumentParser(description="")
    args.add_argument("--batch_size",default=64)
    args.add_argument("--train_file",default="D:/cnki_1/12NLP/entity_ner/resource/datasets/train.char.bmes")
    args.add_argument("--valid_file",default="D:/cnki_1/12NLP/entity_ner/resource/datasets/test.char.bmes")
    args.add_argument("--test_file",default="D:/cnki_1/12NLP/entity_ner/resource/datasets/test.char.bmes")
    args.add_argument("--lr",default=0.001)
    args.add_argument("--device",default=torch.device("cpu"))
    args.add_argument("--epoches",default=50)
    args.add_argument("--log_step",default=50)
    args.add_argument("--test_step",default=100)
    args.add_argument("--vocab_path",default="D:/cnki_1/12NLP/entity_ner/resource/model//vocab11.txt")
    args.add_argument("--label_path",default="D:/cnki_1/12NLP/entity_ner/resource/model//label.txt")
    args.add_argument("--log_path",default="D:/cnki_1/12NLP/entity_ner/resource/model//logs")
    args.add_argument("--vocab_size",default=222)
    args.add_argument("--label_nums",default=10)
    args.add_argument("--embeding_dim",default=32)
    args.add_argument("--hidden_size",default=32)
    args.add_argument("--num_layers",default=2)

    return args.parse_args()



def main():
    args=get_args()
    processor=MyProcessor()
    vocab=processor.get_vocab(args.vocab_path,args.train_file)
    label2id=processor.get_label(args.label_path,args.train_file)
    collator=MyCollator(vocab,label2id)
    train_loader,valid_loader,test_loader=make_loader(collator,args.train_file,args.valid_file,
                                                      args.test_file,args.batch_size)
    args.vocab_size=len(vocab)
    args.label_nums=len(label2id)

    model=BiLSTMCRF(args)
    path = os.path.join("D:/cnki_1/12NLP/entity_ner/resource/model/model.pt")
    model.load_state_dict(torch.load(path))
    model.to(args.device)
    id2label={v:k for k,v in label2id.items()}
    valid(args, model, test_loader, id2label)




def train(args,model,train_loader,valid_loader,test_loader):
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    writer=SummaryWriter(args.log_path)
    step=0
    best_loss=999
    for epoch in range(args.epoches):
        model.train()
        for id ,batch in enumerate(train_loader):
            batch=[w.to(args.device) for w in batch]
            loss=model(input_ids=batch[0],attention_mask=batch[1],labels=batch[2])[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step+=1
            if id%args.log_step==0:
                logging.info("train epoch:{} id:{} loss:{}".format(epoch,id,loss.item()))
                writer.add_scalar("train_loss",loss.item(),step)
                if loss.item()<best_loss:
                    best_loss=loss.item()
                    path=os.path.join("D:/cnki_1/12NLP/entity_ner/resource/model/model.pt")
                    torch.save(model.state_dict(),path)
    return





def valid(args,model,valid_loader,id2label):
    model.eval()
    with torch.no_grad():
        pred_labels = []
        ori_labels = []
        logits_all=[]
        attention_masks=[]
        label_ids=[]
        for id,batch in enumerate(valid_loader):
            batch = [w.to(args.device) for w in batch]
            logits = model(input_ids=batch[0], attention_mask=batch[1])[0]
            logits_all.extend(logits)
            attention_mask=batch[1].cpu().numpy().tolist()
            label_id=batch[2].cpu().numpy().tolist()

            attention_masks.extend(attention_mask)
            label_ids.extend(label_id)

        assert len(logits_all)==len(attention_masks)
        for i in range(len(logits_all)):
            predict=logits_all[i]
            length=sum(attention_masks[i])
            labels=label_ids[i]
            pred_labels.append([id2label[predict[idx]] for idx in range(length)][1:-1])
            ori_labels.append([id2label[labels[idx]] for idx in range(length)][1:-1])
        logging.info(pred_labels)
        logging.info(ori_labels)
        metrics = Metrics(ori_labels, pred_labels, remove_O=False)
        metrics.report_scores()
        metrics.report_confusion_matrix()
    return


def batchPredict(args,model,valid_loader,id2label,id2vocab):
    model.eval()
    with torch.no_grad():
        pred_labels = []
        inputs_ids = []
        logits_all = []
        attention_masks = []
        inputs_vocab = []
        for id, batch in enumerate(valid_loader):
            batch = [w.to(args.device) for w in batch]
            logits = model(input_ids=batch[0], attention_mask=batch[1])[0]
            logits_all.extend(logits)
            attention_mask = batch[1].cpu().numpy().tolist()
            inputs_id=batch[0].cpu().numpy().tolist()

            attention_masks.extend(attention_mask)
            inputs_ids.extend(inputs_id)

        assert len(logits_all) == len(attention_masks)
        for i in range(len(logits_all)):
            predict = logits_all[i]
            length = sum(attention_masks[i])
            inputs = inputs_ids[i]
            pred_labels.append([id2label[predict[idx]] for idx in range(length)][1:-1])
            inputs_vocab.append([id2vocab[inputs[idx]] for idx in range(length)][1:-1])
        return inputs_vocab,pred_labels


def predict(model,id2label,input_ids,attention_mask):
    model.eval()
    with torch.no_grad():
        pred_labels = []
        logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        for l in logits:
            pred_labels.append([id2label[idx] for idx in l])
        print(pred_labels)


def main1(X):
    args = get_args()
    processor = MyProcessor()
    vocab = processor.get_vocab(args.vocab_path, args.train_file)
    label2id = processor.get_label(args.label_path, args.train_file)
    args.vocab_size=len(vocab)
    args.label_nums=len(label2id)
    model=BiLSTMCRF(args)
    path = os.path.join("D:/cnki_1/12NLP/entity_ner/resource/model/model.pt")
    model.load_state_dict(torch.load(path))
    model.to(args.device)
    _words=[vocab[SOS]]+[vocab.get(j,vocab[UNK]) for j in X]+[vocab[EOS]]
    attention_mask=[1 for i in range(len(_words))]
    input_ids=torch.tensor([_words]).long()
    attention_mask=torch.tensor([attention_mask]).long()
    id2label={v:k for k,v in label2id.items()}
    predict(model,id2label,input_ids,attention_mask)

if __name__ == '__main__':
    main()












