import numpy as np
import torch
import torch.nn as nn
import os
from tensorboardX import SummaryWriter
import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s-%(message)s")


from model.sim_model import ConstrastiveLoss
def train(model,trainIter,validIter,args):
    #criterion=nn.MSELoss()
    #criterion=ConstrastiveLoss()
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    writer=SummaryWriter(args.log_path)
    model.train()
    step=0
    total_acc=0
    for epoch in range(1,args.epoches+1):
        for batch in trainIter:
            step+=1
            x1,x2,label=batch
            optimizer.zero_grad()
            logits=model(x1,x2)
            loss=criterion(logits,label)
            loss.backward()
            optimizer.step()
            if step%args.log_interval==0:
                # out=torch.LongTensor(np.array([1 if cos>args.margin else 0 for cos in logits]))
                # out1 = torch.LongTensor(np.array([1 if cos > 0.3 else 0 for cos in logits]))
                # total=out.size(0)
                # acc=(out==label).sum().item()/total
                correct=(torch.max(logits,1)[1].view(label.size()).data==label.data).sum()
                acc=100*correct/label.size(0)
                writer.add_scalar("train_acc",acc)
                writer.add_scalar("train_loss",loss.item())
                logging.info("epoch {} step {} loss:{} acc:{}".format(epoch,step,loss.item(),acc))
            if step%args.test_interval==0:
                loss,acc1=eval_(model,validIter,args)
                if acc1>total_acc:
                    total_acc=acc1
                    logging.info("valid loss:{} acc:{}".format(loss.item(),acc1))
                    logging.info("模型保存...")
                    save_model(model,"G:\software_py\pytorch\孪生神经网络\modelout/model",100)
    writer.close()

def eval_(model,validIter,args):
    #criterion=nn.MSELoss()
    #criterion = ConstrastiveLoss()
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_mean=np.array([],dtype=float)
    correct=0
    total=0
    with torch.no_grad():
        for batch in validIter:
            x1,x2,label=batch
            logits=model(x1,x2)
            loss=criterion(logits,label)
            loss=loss.data.cpu().numpy()
            loss_mean=np.append(loss_mean,loss)
            #out=torch.LongTensor(np.array([1 if cos>args.margin else 0 for cos in logits]))
            #correct+=(out==label).sum().item()
            correct= correct+(torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
            #acc = correct / label.size(0)

            total+=logits.size(0)
    res=np.mean(loss_mean)
    acc=100*correct/total
    logging.info("eval acc:{}".format(acc))
    return res,acc

def save_model(model,save_dir,step):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix=os.path.join(save_dir,"best")
    path=save_prefix+"step{}.bt".format(step)
    torch.save(model.state_dict(),path)
    logging.info("path:{}".format(path))
    return

def load_model(model,path,map_location):
    if os.path.exists(path)==True:
        checkpoint=torch.load(path,map_location=map_location)
        model.load_state_dict(checkpoint)
        logging.info("模型加载成功")
        return model
    else:
        logging.info("无模型可加载")
        return None


