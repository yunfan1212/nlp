
from transformers import BertTokenizer,Trainer,TrainingArguments,default_data_collator
from typing import Optional,List
import datasets
from dataclasses import dataclass,field
from transformers import HfArgumentParser
import torch.nn as nn
from entity_links.el_model import BertForEntityLinking,BertForEntityLinking1
import torch
import logging
import os



'''torch 模型进行训练'''

@dataclass
class DataTrainArgument:
    dataset_name:Optional[str]=field(default="")
    trainfile:Optional[str]=field(default="/home/neurai_project/resources/ELCCKS2009/EL_train.csv")
    testfile: Optional[str] = field(default="/home/neurai_project/resources/ELCCKS2009/EL_test.csv")
    cache_dir:Optional[str]=field(default=None)
    max_length:int=field(default=256)
    preprocessing_num_workers:Optional[int]=field(default=10)


@dataclass()
class ModelArgument():
    model_name_or_path:Optional[str]=field(default="../chinese-bert-wwm-ext/")
    num_tags:int=field(default=2)

def main():

    parser=HfArgumentParser((DataTrainArgument,ModelArgument,TrainingArguments))
    dataset_args,model_args,training_args=parser.parse_args_into_dataclasses()
    print(training_args)
    ###########################加载数据集##############################
    tokenizer=BertTokenizer("../chinese-bert-wwm-ext/vocab.txt")
    trainsets=datasets.load_dataset("csv",data_files={"train":dataset_args.trainfile},cache_dir=dataset_args.cache_dir)
    testsets=datasets.load_dataset("csv",data_files={"test":dataset_args.testfile},cache_dir=dataset_args.cache_dir)

    def preprocess_functions(examples):
        max_length=dataset_args.max_length
        model_inputs={"input_ids":[],"attention_mask":[],"token_type_ids":[],"entity_ids":[],"seq_label":[]}
        texts_list=examples["text"]
        seq_label_list=examples["seq_label"]
        entity_name_list=examples["entity_name"]
        entity_start_list=examples["entity_start"]
        entity_end_list=examples["entity_end"]

        for i in range(len(texts_list)):
            raw_text=texts_list[i]
            seq_label=int(seq_label_list[i])
            entity_name=entity_name_list[i]
            entity_start=entity_start_list[i]
            entity_end=entity_end_list[i]
            text_a,text_b=raw_text.split("#;#")
            tokens_a = tokenizer.tokenize(text_a)
            seq_final_label=[0,0]
            if seq_label==0:
                seq_final_label[0]=1
            else:
                seq_final_label[1]=1
            tokenizer_pre=tokenizer.tokenize(text_b[:entity_start])
            tokenizer_entity=tokenizer.tokenize(entity_name)
            tokenizer_post= tokenizer.tokenize(text_b[entity_end+1:])
            real_label_start=len(tokenizer_pre)
            real_label_end=real_label_start+len(tokenizer_entity)
            tokens_b=tokenizer_pre+tokenizer_entity+tokenizer_post
            #标注

            encoder_dict=tokenizer.encode_plus(tokens_a,text_pair=tokens_b,max_length=max_length,truncation="only_first",padding="max_length",return_token_type_ids=True,
                                            return_attention_mask=True)

            input_ids = encoder_dict["input_ids"]
            token_type_ids = encoder_dict["token_type_ids"]
            attention_mask = encoder_dict["attention_mask"]

            offset=token_type_ids.index(1)
            entity_ids=[0]*max_length
            start_id=offset+real_label_start
            end_id=offset+real_label_end

            if end_id>max_length:
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
    print(preprocess_functions(testsets["test"][:2]))

    #preprocess_functions(trainsets["train"])

    train_datasets=trainsets["train"].map(preprocess_functions,batched=True,
                                 num_proc=dataset_args.preprocessing_num_workers,
                                remove_columns=trainsets["train"].column_names
                                 )
    test_datasets=testsets["test"].map(preprocess_functions,batched=True,
                                 num_proc=dataset_args.preprocessing_num_workers,
                                       remove_columns=testsets["test"].column_names
                                 )
    #######################################加载模型##################################################
    model=BertForEntityLinking1(model_args)
    model=model.cuda()
    print(model)
    model.is_parallelizable = True  # 支持并行
    model.model_parallel = True
    ############################训练###################################
    #训练
    import numpy as np
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support
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
        train_dataset=train_datasets,
        eval_dataset=test_datasets ,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )
    checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)    #加上checkpoint 则为追加训练

    trainer.evaluate()                                         #评估test_datasets
    metrics=train_result.metrics
    print(metrics)

    trainer.save_model(training_args.output_dir)               # Saves the tokenizer too for easy upload, 将最后训练的模型保存到output_dir 文件夹，包括： 模型以及其他文件
    #model.save_pretrained(training_args.output_dir)           #只保存模型,该方法为模型实现,需要自己实现

    ##########################单独评估############################
    # checkpoint = torch.load("./output/pytorch_model.bin")
    # model.load_state_dict(checkpoint)  # 加载模型
    # trainer = Trainer(model=model,compute_metrics=compute_metrics)
    # trainer.evaluate(test_datasets)               #执行完最后打印
    #
    # #####################评估预测############################
    # checkpoint=torch.load("./output/pytorch_model.bin")
    # model.load_state_dict(checkpoint)    #加载模型
    # trainer = Trainer(model=model)
    # trainer.predict(test_datasets)


if __name__ == '__main__':
    main()














