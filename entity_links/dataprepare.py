import json
import random
import pandas as pd

'''数据预处理'''
class ELprocessor():
    def __init__(self):
        '''读取实体-id 信息'''
        with open("../resources/ccks2019/entity_to_ids.json","r",encoding="utf8") as f:
            self.entity_to_ids=json.loads(f.read())
        '''读取id-描述信息'''
        with open("../resources/ccks2019/subject_id_with_info.json","r",encoding="utf8") as f:
            self.subject_id_with_info=json.loads(f.read())
    '''读取训练集'''
    def read_json(self,path):
        with open(path,"r",encoding="utf8") as f:
            lines=f.readlines()
        return lines
    '''数据打标，获取训练数据'''
    def build_datasets(self,lines):
        examples=[]
        for i,line in enumerate(lines):
            line=eval(line)
            text=line["text"].lower()
            if len(text)<2:
                continue
            for mention_data in line["mention_data"]:
                word=mention_data["mention"].lower()
                if len(word)<=1:
                    continue
                kb_id=mention_data["kb_id"]
                start_id=int(mention_data["offset"])
                end_id=start_id+len(word)-1
                '''获取实体相关的描述'''
                rel_texts=self.get_text_pair(word,kb_id,text)
                for i,rel_text in enumerate(rel_texts):
                    if i==0:
                        examples.append({"text":rel_text,"seq_label":1,"entity_id":kb_id,"entity_name":word,"entity_start":start_id,"entity_end":end_id})
                    else:
                        examples.append(
                            {"text": rel_text, "seq_label": 0, "entity_id":kb_id,"entity_name":word,"entity_start":start_id,"entity_end":end_id})
        return examples

    def get_text_pair(self,word,kb_id,text):
        """
        用于构建正负样本对，一个正样本，三个负样本
        :return:
        """
        results=[]
        if kb_id!='NIL' and word in self.entity_to_ids:
            text_a=self.get_info(kb_id)
            if len(text_a)<5:
                return []
            pos_example=text_a+"#;#"+text
            results.append(pos_example)
            ids=self.entity_to_ids[word]
            if "NIL" in ids:
                ids.remove("NIL")
            ind=ids.index(kb_id)
            ids=ids[:ind]+ids[ind+1:]
            if len(ids)>=3:
                ids=random.sample(ids,3)
            for t_id in ids:
                text_a = self.get_info(t_id)
                if len(text_a) < 5:
                    continue
                neg_example=text_a+"#;#"+text
                results.append(neg_example)
        return results
    def get_info(self,subject_id):
        '''
        找到文本描述
        :param subject_id:
        :return:
        '''
        infos=self.subject_id_with_info[subject_id]
        data=infos["data"]
        res=[]
        for kg in data:
            if kg["object"][-1]!="。":
                res.append("{}.{}。".format(kg["predicate"],kg["object"]))
            else:
                res.append("{}.{}".format(kg["predicate"],kg["object"]))

        return "".join(res).lower()
if __name__ == '__main__':
    '''数据预处理，将源数据进行整理，获取训练集、测试集'''
    elprocessor=ELprocessor()
    datasets=elprocessor.read_json("../resources/ccks2019/train.json")
    raw_examples=elprocessor.build_datasets(datasets)
    total_num=len(raw_examples)
    print(total_num)
    train_total=int(total_num*0.8)
    test_total=total_num-train_total
    print(f"数据集总共{total_num},训练集数量{train_total} 测试集数量{test_total}")
    random.shuffle(raw_examples)
    train_examples=raw_examples[:train_total]
    test_examples=raw_examples[train_total:]
    print(train_examples[:2])
    train_dfs=pd.DataFrame(train_examples)
    test_dfs=pd.DataFrame(test_examples)
    train_dfs.to_csv("../resources/EL_train.csv")
    test_dfs.to_csv("../resources/EL_test.csv")



