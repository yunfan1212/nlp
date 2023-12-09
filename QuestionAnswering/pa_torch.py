
import sys
from datasets import load_dataset

import numpy as np
import re
import nltk
'''阅读理解'''
import collections
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, default_data_collator, TrainingArguments, Trainer

def main():

    #加载数据集##################################
    raw_datasets=load_dataset("cmrc2018")
    tokenizer=AutoTokenizer.from_pretrained("../rbt3")

    # 数据集处理
    max_length = 384
    doc_stride = 128

    def process_function_for_train(examples):
        examples["question"] = [q.strip() for q in examples["question"]]
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            max_length=max_length,
            truncation="only_second",  # 指定改参数，将只在第二部分输入上进行截断，即文章部分进行截断
            return_overflowing_tokens=True,  # 指定该参数，会根据最大长度与步长将恩本划分为多个段落
            return_offsets_mapping=True,  # 指定改参数，返回切分后的token在文章中的位置
            stride=doc_stride,
            padding="max_length"
        )
        # 对于阅读理解任务，标签数据不再是labels, 而是start_positions和end_positions，分别存储起始和结束位置
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["example_id"] = []
        # sample_mapping中存储着新的片段对应的原始example的id，例如[0, 0, 0, 1, 1, 2]，表示前三个片段都是第1个example
        # 根据sample_mapping中的映射信息，可以有效的定位答案
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        for i, _ in enumerate(sample_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            answers = examples["answers"][sample_mapping[i]]  # 根据sample_mapping的结果，获取答案的内容
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 定位文章的起始token位置
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # 定位文章的结束token位置
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            offsets = tokenized_examples["offset_mapping"][i]

            # 判断答案是否在当前的片段里，条件：文章起始token在原文中的位置要小于答案的起始位置，结束token在原文中的位置要大于答案的结束位置
            # 如果不满足，则将起始与结束位置均置为0
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(0)
                tokenized_examples["end_positions"].append(0)
            else:  # 如果满足，则将答案定位到token的位置上
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

            # 定位答案相关
            tokenized_examples["example_id"].append(examples["id"][sample_mapping[i]])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

   # process_function_for_train(raw_datasets["train"][:2])

    def get_result(start_logits, end_logits, examples, features):

        predicted_answers = {}
        reference_answers = {}

        # 构建example到feature的映射
        example_to_features = collections.defaultdict(list)
        for idx, feature_id in enumerate(features["example_id"]):
            example_to_features[feature_id].append(idx)       #{样本id:[索引号]}

        # 指定备选最优答案个数与最大答案长度
        n_best = 20
        max_answer_length = 30

        # 抽取答案
        for example in examples:          #每个输入样本,未拆分的样本
            example_id = example["id"]           # 样本id
            context = example["context"]        #context
            answers = []
            # 对当前example对应的所有feature片段进行答案抽取
            for feature_index in example_to_features[example_id]:   #样本id  对应的特征id 列表
                start_logit = start_logits[feature_index]            #特征 开始的预测
                end_logit = end_logits[feature_index]                #特征 结束的预测
                offsets = features[feature_index]["offset_mapping"]   # 特征起始位置对应表

                start_indexes = np.argsort(start_logit)[:: -1][:n_best].tolist()   #开始的index，取前20个
                end_indexes = np.argsort(end_logit)[:: -1][:n_best].tolist()       #结束的index  取前20个

                for start_index in start_indexes:   #开始的索引列表
                    for end_index in end_indexes:   #结束的索引列表
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        if (end_index < start_index or end_index - start_index + 1 > max_answer_length):
                            continue
                        answers.append(           #抽取文本+得分      满足要求的字段 ；得分
                            {
                                "text": context[offsets[start_index][0]: offsets[end_index][1]],        #每一个样本抽取的特征
                                "logit_score": start_logit[start_index] + end_logit[end_index],         #每一个样本抽取特征的得分
                            }
                        )
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])    #抽取结果按得分排序
                predicted_answers[example_id] = best_answer["text"]           #抽取的结果
            else:
                predicted_answers[example_id] = ""
            reference_answers[example_id] = example["answers"]["text"]      #获取答案

        return predicted_answers, reference_answers

    train_datasets=raw_datasets["train"]#.select([1,2,3,4,5])
    valid_datasets=raw_datasets["validation"]#.select([1])

    tokenized_train_dataset = train_datasets.map(process_function_for_train, batched=True,num_proc=10,
                                                     remove_columns=raw_datasets["train"].column_names)

    tokenized_valid_dataset = valid_datasets.map(process_function_for_train, batched=True,num_proc=10,
                                                         remove_columns=raw_datasets["validation"].column_names)
    tokenized_test_dataset = raw_datasets["test"].map(process_function_for_train, batched=True,
                                                  remove_columns=raw_datasets["test"].column_names,num_proc=10)

    def mixed_segmentation(in_str, rm_punc=False):
        in_str = str(in_str).lower().strip()
        segs_out = []
        temp_str = ""
        sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                   '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                   '「', '」', '（', '）', '－', '～', '『', '』']
        for char in in_str:
            if rm_punc and char in sp_char:
                continue
            if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
                if temp_str != "":
                    ss = nltk.word_tokenize(temp_str)
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char

        # handling last part
        if temp_str != "":
            ss = nltk.word_tokenize(temp_str)
            segs_out.extend(ss)

        return segs_out

    # remove punctuation
    def remove_punctuation(in_str):
        in_str = str(in_str).lower().strip()
        sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                   '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                   '「', '」', '（', '）', '－', '～', '『', '』']
        out_segs = []
        for char in in_str:
            if char in sp_char:
                continue
            else:
                out_segs.append(char)
        return ''.join(out_segs)


    # find longest common string
    def find_lcs(s1, s2):
        m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
        mmax = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        return s1[p - mmax:p], mmax

    def calc_f1_score(answers, prediction):
        f1_scores = []
        for ans in answers:
            ans_segs = mixed_segmentation(ans, rm_punc=True)
            prediction_segs = mixed_segmentation(prediction, rm_punc=True)
            lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
            if lcs_len == 0:
                f1_scores.append(0)
                continue
            precision = 1.0 * lcs_len / len(prediction_segs)
            recall = 1.0 * lcs_len / len(ans_segs)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)
        return max(f1_scores)

    def calc_em_score(answers, prediction):
        em = 0
        for ans in answers:
            ans_ = remove_punctuation(ans)
            prediction_ = remove_punctuation(prediction)
            if ans_ == prediction_:
                em = 1
                break
        return em
    def evaluate_cmrc(predictions, references):
        f1 = 0
        em = 0
        total_count = 0
        skip_count = 0
        for query_id, answers in references.items():
            total_count += 1
            if query_id not in predictions:
                sys.stderr.write('Unanswered question: {}\n'.format(query_id))
                skip_count += 1
                continue
            prediction = predictions[query_id]
            f1 += calc_f1_score(answers, prediction)
            em += calc_em_score(answers, prediction)
        f1_score = 100.0 * f1 / total_count
        em_score = 100.0 * em / total_count
        return {
            'avg': (em_score + f1_score) * 0.5,
            'f1': f1_score,
            'em': em_score,
            'total': total_count,
            'skip': skip_count
        }


    def compute_metrics(p):
        start_logits, end_logits = p[0]
        if start_logits.shape[0] == len(tokenized_valid_dataset):
            predicted_answers, reference_answers = get_result(start_logits, end_logits, valid_datasets,
                                                              tokenized_valid_dataset)
        else:
            predicted_answers, reference_answers = get_result(start_logits, end_logits, raw_datasets["test"],
                                                              tokenized_test_dataset)
        return evaluate_cmrc(predicted_answers, reference_answers)


    model = AutoModelForQuestionAnswering.from_pretrained("../rbt3")
    args = TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        output_dir="model_for_qa",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="avg",
        fp16=True
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator = default_data_collator)
    trainer.train()
    trainer.evaluate(tokenized_test_dataset)


if __name__ == '__main__':
    main()




