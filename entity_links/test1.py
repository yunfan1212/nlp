
import logging

from neurai.config import set_platform
set_platform(platform="gpu")
from neurai.datasets import Dataset,DataLoader
import argparse
from transformers import BertTokenizer
import datasets
import neurai.nn as nn
import jax.numpy as nnp
import neurai.initializer as initializer
from neurai.nn.layer import RNN,Linear,LSTMCell
import optax as opt
from tqdm import tqdm
from jax import jit
from neurai.nn.layer.loss import softmax_cross_entropy
import jax




logging.basicConfig(level=logging.INFO,format="%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s")
def get_args():
    parser=argparse.ArgumentParser(description="el")
    parser.add_argument("--testfile",default="/home/jiayafei_linux/neurai_project/resources/ELCCKS2009/EL_test.csv")
    parser.add_argument("--trainfile", default="/home/jiayafei_linux/neurai_project/resources/ELCCKS2009/EL_train.csv")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_length", default=256)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--hidden_size", default=768)
    parser.add_argument("--lr", default=1e-6)
    parser.add_argument("--log_step", default=50)
    parser.add_argument("--vocab_size", default=21128)
    parser.add_argument("--initializer_range", default=0.02)
    parser.add_argument("--num_tags", default=2)
    parser.add_argument("--num_epoch", default=50)
    parser.add_argument("--preprocessing_num_workers", default=10)

    return parser.parse_args()


class MyDataset(Dataset):
    def __init__(self,dataset):
        self.data=dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]


class DefaltCollactor():
    def __call__(self, *args,**kwargs):
        batch=args[0]
        columns=list(batch[0].keys())
        res={w:[]for w in columns}
        for w in batch:
            for _c in columns:
                res[_c].append(w[_c])
        res={k:nnp.asarray(v,nnp.int32) for k,v in res.items()}
        return res





class ELLinkModel(nn.Module):
    config=get_args()
    dtype:nnp.dtype=nnp.float32
    def setup(self):
        self.embed=nn.Embed(self.config.vocab_size,self.config.hidden_size,
                            embedding_init=initializer.NormalIniter(stddev=self.config.initializer_range)
                            ,dtype=self.dtype,
                            name='word embedding'
                            )
        self.lstm=RNN(LSTMCell(self.config.hidden_size,768,kernel_init=initializer.NormalIniter(stddev=self.config.initializer_range) ))
        self.lstm1 = RNN(LSTMCell(768, 768,kernel_init=initializer.NormalIniter(stddev=self.config.initializer_range)))
        self.pool=nn.Linear(768)
        #self.pool1 = nn.Linear(768)
        self.linear=nn.Linear(self.config.num_tags)


    def __call__(self, input_ids, attention_mask, token_type_ids, seq_label, entity_ids,train:bool = True):

        embedding=self.embed(input_ids.astype("int32"))
        state,output=self.lstm(embedding)                 #ouput:[batch,seq_len,256]
        state, output = self.lstm1(output)
        pool=self.pool(output.reshape(output.shape[0],-1))             #[batch,256]

        token_out=output     #[batch,seq_len,hidden]
        seq_out=pool            #[batch.hidden]
        batch_out=[]
        for t_out,t_mask,s_out in zip(token_out,entity_ids,seq_out):
            t_mask=t_mask==1
            mask = t_mask.astype(nnp.int32)[:,None]
            entity_out = t_out * mask

            out = nnp.concatenate([entity_out, s_out[nnp.newaxis, :]], axis=0)[None, :, :, None]  # [1,257,hidden]

            out = nn.layer.MaxPool(window_shape=(self.config.max_length + 1, 1),
                                   strides=(self.config.max_length + 1, 1), padding="VALID")(out)
            # out=self.pool1(out.reshape(1,-1))

            out = out.reshape(1, -1)
            assert out.shape[1] == 768
            batch_out.append(out)
        batch_out = nnp.concatenate(batch_out, axis=0)
        logits = self.linear(batch_out)

        return logits



def accuracy(predict,target):
    return nnp.mean(nnp.argmax(predict, axis=1) == nnp.argmax(target, axis=1))


def main():

    args=get_args()

    #�~J| 载�~U��~M��~[~F
    tokenizer=BertTokenizer.from_pretrained("../chinese-bert-wwm-ext")
    raw_datasets=datasets.load_dataset("csv",data_files={"train":args.trainfile},cache_dir=args.cache_dir).remove_columns(["Unnamed: 0"])
    raw_datasets=raw_datasets["train"].train_test_split(test_size=0.2)
    train_datasets=raw_datasets["train"].shuffle()
    test_datasets=raw_datasets["test"].select(list(range(1000)))
    #print(train_datasets)

    def preprocess_functions(examples):
        max_length = args.max_length
        model_inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "entity_ids": [], "seq_label": []}
        texts_list = examples["text"]
        seq_label_list = examples["seq_label"]
        entity_name_list = examples["entity_name"]
        entity_start_list = examples["entity_start"]
        entity_end_list = examples["entity_end"]
        for i in range(len(texts_list)):
            raw_text = texts_list[i]
            seq_label = int(seq_label_list[i])
            entity_name = entity_name_list[i]
            entity_start = entity_start_list[i]
            entity_end = entity_end_list[i]
            text_a, text_b = raw_text.split("#;#")

            # 分词
            tokens_a = tokenizer.tokenize(text_a)
            seq_final_label = [0, 0]
            if seq_label == 0:
                seq_final_label[0] = 1
            else:
                seq_final_label[1] = 1
            tokenizer_pre = tokenizer.tokenize(text_b[:entity_start])
            tokenizer_entity = tokenizer.tokenize(entity_name)
            tokenizer_post = tokenizer.tokenize(text_b[entity_end + 1:])
            real_label_start = len(tokenizer_pre)
            real_label_end = real_label_start + len(tokenizer_entity)
            tokens_b = tokenizer_pre + tokenizer_entity + tokenizer_post

            # 标注
            encoder_dict = tokenizer.encode_plus(tokens_a, text_pair=tokens_b, max_length=max_length,
                                                 truncation="only_first",
                                                 padding="max_length", return_token_type_ids=True,
                                                 return_attention_mask=True)
            input_ids = encoder_dict["input_ids"]
            token_type_ids = encoder_dict["token_type_ids"]
            attention_mask = encoder_dict["attention_mask"]

            offset = token_type_ids.index(1)
            entity_ids = [0] * max_length
            start_id = offset + real_label_start
            end_id = offset + real_label_end
            if end_id > max_length:
                print('发生了不该有的截断')
                for i in range(start_id, max_length):
                    entity_ids[i] = 1
            else:
                for i in range(start_id, end_id):
                    entity_ids[i] = 1
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["token_type_ids"].append(token_type_ids)
            model_inputs["entity_ids"].append(entity_ids)
            model_inputs["seq_label"].append(seq_final_label)
        return model_inputs

    train_datasets_ = train_datasets.map(preprocess_functions, batched=True,
                                         num_proc=args.preprocessing_num_workers,
                                         remove_columns=train_datasets.column_names)
    test_datasets_ = test_datasets.map(preprocess_functions, batched=True,
                                       num_proc=args.preprocessing_num_workers,
                                       remove_columns=train_datasets.column_names)

    # #创建加载器
    train_loads = DataLoader(MyDataset(train_datasets_), batch_size=args.batch_size, shuffle=True, drop_last=True,
                             collate_fn=DefaltCollactor())
    test_loads = DataLoader(MyDataset(test_datasets_), batch_size=args.batch_size, shuffle=False,
                            collate_fn=DefaltCollactor())











