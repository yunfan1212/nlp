

import datasets
from transformers import HfArgumentParser,TrainingArguments,Trainer
from transformers import AutoTokenizer,AutoModelForTokenClassification,DataCollatorForTokenClassification
import numpy as np
from dataclasses import dataclass,field

@dataclass()
class DataArgument():
    pass
'''实体识别，torch'''

def main():

    #########################数据集加载#######################
    # raw_dataset=datasets.load_dataset("peoples_daily_ner")
    # raw_dataset.save_to_disk("people_daily_ner")
    raw_dataset=datasets.load_from_disk("people_daily_ner")
    print(raw_dataset["train"][:2])
    tokenizer=AutoTokenizer.from_pretrained("../chinese-bert-wwm-ext")

    def process_function(examples):
        tokenized_examples = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=64)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_examples.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            labels.append(label_ids)
        tokenized_examples["labels"] = labels
        return tokenized_examples

    tokenized_datasets = raw_dataset.map(process_function, batched=True)
    label_list = tokenized_datasets["train"].features["ner_tags"].feature.names
    print(label_list)
    ###########################加载模型#####################################
    model = AutoModelForTokenClassification.from_pretrained("../chinese-bert-wwm-ext", num_labels=len(label_list))
    model=model.cuda()
    ###############################训练#######################################################
    #seqeval_metric = evaluate.load("seqeval")             #pip install seqeval
    from seqeval.scheme import IOB2
    from seqeval.metrics import classification_report
    from seqeval.metrics import f1_score,precision_score, recall_score,accuracy_score

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        f1=f1_score(true_labels, true_predictions)                   #一维数据
        acc=accuracy_score(true_labels, true_predictions)
        recall=recall_score(true_labels, true_predictions)
        precision=precision_score(true_labels, true_predictions)
        # results = seqeval_metric.compute(predictions=true_predictions, references=true_labels, mode="strict",
        #                                  scheme="IOB2")
        res = classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2)
        print(res)
        print({'accuracy':acc, 'f1': f1, 'precision': precision, 'recall':recall})
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    args = TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        weight_decay=0.01,
        output_dir="output",
        logging_steps=10,
        evaluation_strategy="epoch",            #评估策略
        save_strategy="epoch",                   #保存策略
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # 训练与评估
    trainer.train()
    print("====================测试=========================")
    trainer.evaluate(tokenized_datasets["test"])
    trainer.save_model(args.output_dir)


    ############################加载trainer并评估#########################################
    import torch
    checkpoint=torch.load("./output/pytorch_model.bin")
    model.load_state_dict(checkpoint)
    args = TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        weight_decay=0.01,
        output_dir="output",
        logging_steps=10,
        evaluation_strategy="epoch",            #评估策略
        save_strategy="epoch",                   #保存策略
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.evaluate(tokenized_datasets["test"])

    # import torch
    #
    # model.to('cpu')
    # text=""
    # inputs = tokenizer(text, return_tensors="pt")
    # with torch.no_grad():
    #     logits = model(**inputs).logits
    #     predictions = torch.argmax(logits, dim=2)
    # predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
    # print(predicted_token_class)




if __name__ == '__main__':
    main()












