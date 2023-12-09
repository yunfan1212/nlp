#coding=utf8
import torch
from transformers import HfArgumentParser
from dataclasses import field,dataclass
from datasets import load_dataset
from typing import Optional,List
from disambiguation.dis_model import Disambiguation

from transformers import (TrainingArguments,Trainer,default_data_collator,
                          BertTokenizer)
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support


'''指代消歧'''
@dataclass()
class DataTrainArgument():
    trainfile:Optional[str]=field(default="/home/neurai_project/resources/cluewsc2020/train.json")
    testfile:Optional[str]=field(default="/home/neurai_project/resources/cluewsc2020/dev.json")
    max_length:Optional[int]=field(default=256)
    preprocessing_num_workers:Optional[int]=field(default=10)
    cache_dir:Optional[str]=field(default=None)

@dataclass
class ModelArgument:
    model_name_or_path: Optional[str] = field(default="../chinese-bert-wwm-ext/")
    num_tags:int=field(default=2)


def main():
    #*************************加载数据*******************************************
    parser=HfArgumentParser((DataTrainArgument,ModelArgument,TrainingArguments))
    dataset_args,model_args,training_args=parser.parse_args_into_dataclasses()

    #数据集加载
    tokenizer=BertTokenizer.from_pretrained(model_args.model_name_or_path)
    trainsets=load_dataset("json",data_files={"train":dataset_args.trainfile},cache_dir=dataset_args.cache_dir).shuffle()
    testsets=load_dataset("json",data_files={"test":dataset_args.testfile},cache_dir=dataset_args.cache_dir)
    print(trainsets)
    print(testsets)

    max_seq_length=dataset_args.max_length

    def preprocess_function(examples):
        '''数据处理'''
        texts=examples["text"]
        targets=examples["target"]
        labels=examples["label"]

        model_inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "span1_ids": [],"span2_ids": [], "seq_label": []}
        assert len(texts)==len(targets)==len(labels)
        for i in range(len(texts)):
            sub_text=texts[i]
            target=targets[i]
            span1=(target["span1_text"],target["span1_index"])
            span2=(target["span2_text"],target["span2_index"])
            label=labels[i]

            seq_final_label=[0,0]
            assert label in ["true","false"]
            if label=="false":
                seq_final_label[0]=1
            else:
                seq_final_label[1]=1

            #标注
            if span1[1]<span2[1]:  #对齐

                tokenizer_pre=tokenizer.tokenize(sub_text[:span1[1]])
                tokenizer_span1=tokenizer.tokenize(span1[0])
                tokenizer_mid=tokenizer.tokenize(sub_text[span1[1]+len(span1[0]):span2[1]])
                tokenier_span2=tokenizer.tokenize(span2[0])
                tokenizer_post=tokenizer.tokenize(sub_text[span2[1]+len(span2[0]):])
                tokens=tokenizer_pre+tokenizer_span1+tokenizer_mid+tokenier_span2+tokenizer_post
                real_span1_start=len(tokenizer_pre)
                real_span1_end=real_span1_start+len(tokenizer_span1)

                real_span2_start=len(tokenizer_pre)+len(tokenizer_span1)+len(tokenizer_mid)
                real_span2_end=real_span2_start+len(tokenier_span2)

            else:

                tokenizer_pre = tokenizer.tokenize(sub_text[:span2[1]])
                tokenizer_span2 = tokenizer.tokenize(span2[0])
                tokenizer_mid = tokenizer.tokenize(sub_text[span2[1] + len(span2[0]):span1[1]])
                tokenier_span1 = tokenizer.tokenize(span1[0])
                tokenizer_post = tokenizer.tokenize(sub_text[span1[1] + len(span1[0]):])
                tokens = tokenizer_pre + tokenizer_span2 + tokenizer_mid + tokenier_span1 + tokenizer_post

                real_span2_start = len(tokenizer_pre)
                real_span2_end = real_span2_start + len(tokenizer_span2)
                real_span1_start = len(tokenizer_pre) + len(tokenizer_span2) + len(tokenizer_mid)
                real_span1_end = real_span1_start + len(tokenier_span1)

            span1_ids=[tokenizer.pad_token_id]*len(tokens)

            for i in range(real_span1_start,real_span1_end):
                span1_ids[i]=1
            span2_ids=[tokenizer.pad_token_id]*len(tokens)
            for i in range(real_span2_start,real_span2_end):
                span2_ids[i]=1
            if len(span1_ids)<=max_seq_length-2:
                pad_length=max_seq_length-2-len(span1_ids)
                span1_ids=span1_ids+[tokenizer.pad_token_id]*pad_length
                span2_ids=span2_ids+[tokenizer.pad_token_id]*pad_length
                span1_ids=[tokenizer.cls_token_id]+span1_ids+[tokenizer.sep_token_id]
                span2_ids=[tokenizer.cls_token_id]+span2_ids+[tokenizer.sep_token_id]
            else:
                if real_span2_end>max_seq_length-2:
                    return "发生不该有的截断"
                span1_ids=span1_ids[:max_seq_length-2]
                span2_ids=span2_ids[:max_seq_length-2]
                span1_ids = [tokenizer.cls_token_id] + span1_ids + [tokenizer.sep_token_id]
                span2_ids = [tokenizer.cls_token_id] + span2_ids + [tokenizer.sep_token_id]

            assert len(span1_ids) == max_seq_length
            assert len(span2_ids) == max_seq_length

            encode_dict=tokenizer.encode_plus(text=tokens,max_length=max_seq_length,
                                               padding="max_length",
                                               truncation="only_first",
                                               return_token_type_ids=True,
                                               return_attention_mask=True)
            input_ids = encode_dict['input_ids']
            attention_mask = encode_dict['attention_mask']
            token_type_ids = encode_dict['token_type_ids']

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["token_type_ids"].append(token_type_ids)
            model_inputs["seq_label"].append(seq_final_label)
            model_inputs["span1_ids"].append(span1_ids)
            model_inputs["span2_ids"].append(span2_ids)
        return model_inputs

    trainsets=trainsets["train"].map(preprocess_function,batched=True,num_proc=dataset_args.preprocessing_num_workers,
                                     remove_columns=trainsets["train"].column_names)
    testsets = testsets["test"].map(preprocess_function, batched=True,
                                       num_proc=dataset_args.preprocessing_num_workers,
                                       remove_columns=testsets["test"].column_names)


    ###########加载模型
    model=Disambiguation(model_args)
    model.to(torch.device("cuda:1"))
    print(model)
    model.is_parallelizable = True  # 支持并行
    model.model_parallel = True

    ############################开始训练
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        preds = np.argmax(preds, axis=1)
        labels=np.argmax(labels, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        print({'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall})
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainsets,
        eval_dataset=testsets ,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )
    checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)    #加上checkpoint 则为追加训练

    trainer.evaluate()                                         #评估test_datasets
    metrics=train_result.metrics
    print(metrics)

    trainer.save_model(training_args.output_dir)


if __name__ == '__main__':
    main()











