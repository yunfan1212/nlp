import os
import torch
import logging
from mysqltool.mysql_pool import MysqlModel
from sentence_classify.cnn_main import args_parse,MyProcessor,MyCollator,CNN_Model
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(levelname)s-%(filename)s-%(message)s")


class Infer():
    def __init__(self):
        args = args_parse()
        self.args=args
        process = MyProcessor()
        vocab = process.make_vocab(args.vocab_path)
        label_vocab = process.get_class_vocab(args.label_path)
        self.id2label={v:k for k,v in label_vocab.items()}
        self.collator = MyCollator(vocab_dict=vocab, label_dict=label_vocab, max_len=args.max_len, min_len=args.min_len)

        self.model = CNN_Model(vocab_size=len(vocab), embed_dim=args.embed_dim, kernels=args.kernels,
                          out_channels=args.out_channels,
                          hidden_dim=args.hidden_dim, out_dim=len(label_vocab), drop_tate=args.drop_rate)
        save_path = os.path.join(args.save_path, "{}_model.pt".format("cls"))
        checkpoint=torch.load(save_path)
        self.model.load_state_dict(checkpoint)
        self.model.to(args.device)

    def predict(self,text_list):
        sent_tensor=self.collator.text_prepare(text_list)
        sent_tensor.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            output=self.model(sent_tensor)
            predict=torch.max(output,dim=-1)[1]
            predict=predict.cpu().data.tolist()
            predict=[self.id2label[id] for id in predict]
        return predict

class DataLabel():
    def __init__(self):
        self.model=Infer()

    def get_data(self):
        logging.info("predict...")
        sql="SELECT count(*) FROM a_znzz_enterprise_product where product_name!='' and is_sample =0"
        count=MysqlModel.query(sql)[0]["count(*)"]
        logging.info("num {}".format(count))
        batch=64
        for i in range(0,count,batch):
            start=i
            sql = "SELECT id,product_desc,product_name FROM a_znzz_enterprise_product where product_name!='' and is_sample =0 limit {},{}".format(start,batch)
            res = MysqlModel.query(sql)
            logging.info(res[:10])
            if len(res)>0:
                text_list=[w["product_name"]+w["product_desc"] for w in res if w["product_name"]+w["product_desc"]!=""]
                result=self.model.predict(text_list)
                insert_data=[]
                assert len(result)==len(res)
                sql_=""
                for i in range(len(result)):
                    insert_data.append([res[i]["id"],result[i]])
                    sql_+="( {},'{}'),".format(res[i]["id"],result[i])
                sql_=sql_[:-1]
                sql="insert into znzz.a_znzz_enterprise_product(id,first_category) values "+sql_+" on duplicate key update first_category=values(first_category);"
                logging.info(sql)
                MysqlModel.update(sql)
        return



if __name__ == '__main__':
    x=DataLabel()
    x.get_data()





