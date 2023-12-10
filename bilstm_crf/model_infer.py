


from entity_ner.bilstm_crf.model_main import get_args,MyCollator,MyProcessor,BiLSTMCRF,os,batchPredict,MyDataSets
import torch
from torch.utils.data import DataLoader


class modelInfer():
    def __init__(self):
        args = get_args()
        self.args=args
        processor = MyProcessor()
        vocab = processor.get_vocab(args.vocab_path, args.train_file)
        label2id = processor.get_label(args.label_path, args.train_file)

        self.collator = MyCollator(vocab, label2id)

        args.vocab_size = len(vocab)
        args.label_nums = len(label2id)

        self.model = BiLSTMCRF(args)
        path = os.path.join("D:/cnki_1/12NLP/entity_ner/resource/model/model.pt")
        self.model.load_state_dict(torch.load(path))
        self.model.to(args.device)
        self.id2label = {v: k for k, v in label2id.items()}
        self.id2vocab={v: k for k, v in vocab.items()}

    def forward(self,tokenslist,labelslists):
        valid_loader = DataLoader(MyDataSets(tokenslist,labelslists), batch_size=64, shuffle=False, num_workers=4,
                                  collate_fn=self.collator)
        sentences,predicts=batchPredict(self.args,self.model,valid_loader,self.id2label,self.id2vocab)
        print(predicts)
        return sentences,predicts










