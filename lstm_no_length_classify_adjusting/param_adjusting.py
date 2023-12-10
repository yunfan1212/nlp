import optuna
from numpy.core.umath_tests import inner1d
from lstm_no_length_classify_adjusting.main import args_parse,MyProcessor,BertTokenizer,MyCollator,make_loader
from lstm_no_length_classify_adjusting.main import LstmModel,train



model_name=[str(i) for i in range(1000)]

def objective(trail):
    params={"lr":trail.suggest_loguniform("lr",1e-3,1e-1),
            "num_layers":trail.suggest_int("batch_size",32,128)}

    args = args_parse()
    args.lr=params["lr"]
    args.num_layers=params["num_layers"]
    args.model_name=model_name.pop(0)
    processor = MyProcessor(args.train_file)
    label2id = processor.get_labels(args.label_path)

    tokenizer = BertTokenizer.from_pretrained("../lstm_classify_model_adjusting/resource/vocab.txt")
    collator = MyCollator(tokenizer, args.max_len, label2id)
    train_loader, eval_loader, test_loader = make_loader(collator, args.train_file, args.eval_file,
                                                         args.test_file, args.batch_size)
    args.vocab_size = len(tokenizer)
    args.label_num = len(label2id)
    model = LstmModel(args)
    model.to(args.device)

    acc = train(model, args, train_loader, eval_loader)
    return acc


if __name__ == '__main__':
    study=optuna.create_study(study_name="test0622",direction="maximize",sampler=optuna.samplers.TPESampler(),
                              storage="sqlite:///db.sqlite3")
    study.optimize(objective,n_trials=30)
    best_trail=study.best_trial
    for k,v in best_trail.items():
        print(k,v)











