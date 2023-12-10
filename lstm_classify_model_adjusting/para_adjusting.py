from numpy.core.umath_tests import inner1d
import optuna
from lstm_classify_model_adjusting.main import args_parse,get_labels,BertTokenizer,Collator,make_loader,LstmClassify
from lstm_classify_model_adjusting.main import train


model_name=[str(i) for i in range(1000)]

def objective(trail):

    params={"lr":trail.suggest_loguniform("lr",1e-4,1e-1),
            "batch_size":trail.suggest_int("batch_size",32,128),
            "num_layers":trail.suggest_int("num_layers",1,3),
            "embed_dim":trail.suggest_int("embed_dim",32,64),
            "hidden_dim":trail.suggest_int("hidden_dim",32,128)}

    args = args_parse()
    args.lr=params["lr"]
    args.batch_size=params["batch_size"]
    args.num_layers=params["num_layers"]
    args.embed_dim=params["embed_dim"]
    args.hidden_dim=params["hidden_dim"]
    args.model_name=model_name.pop(0)

    tokenizer = BertTokenizer("./resource/vocab.txt")
    label_dict = get_labels(args.train_file, args.label_path)
    args.vocab_size = len(tokenizer)
    args.label_nums = len(label_dict)

    collator_fn = Collator(tokenizer, args.max_len, label_dict)
    train_loader, eval_loader, test_loader = make_loader(collator_fn, args.train_file,
                                                         args.eval_file, args.test_file, args.batch_size)

    model = LstmClassify(args)


    acc = train(model, args, train_loader, eval_loader)
    return acc


if __name__ == '__main__':
    study=optuna.create_study(study_name="test0620",direction="maximize",sampler=optuna.samplers.TPESampler(),
                              storage="sqlite:///db.sqlite3")
    study.optimize(objective,n_trials=20)
    best_trail=study.best_trial
    for key,value in best_trail.items():
        print(key,value)


    '''
    optuna-dashboard sqlite:///db.sqlite3 --host=0.0.0.0
    tensorboard --logdir=‘log’
    '''


