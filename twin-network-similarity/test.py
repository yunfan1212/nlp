
import torch
import torch.nn as nn
from torch.nn import CosineEmbeddingLoss
import numpy as np


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom
def cal_score(score, target):
    if target == 1:
        return 1 - score
    else:
        return max(0, score)
#均值损失函数
def criterion_my(x1, x2, target, reduction='mean'):
    batch_size, hidden_size = x1.size()
    scores = torch.cosine_similarity(x1, x2)
    for i in range(batch_size):
        scores[i] = cal_score(scores[i], target[i].item())
    if reduction == 'mean':
        return scores.mean()
    elif reduction == 'sum':
        return scores.sum()
#mse损失函数
def criterion_my_mse(x1,x2,target):
    score=torch.cosine_similarity(x1,x2)
    loss=nn.MSELoss()
    return loss(score,target)
#均值损失函数
def criterion_my2(x1, x2, target, reduction='mean'):
    batch_size, hidden_size = x1.size()
    scores = torch.zeros(batch_size)
    for i in range(batch_size):
        score = cosine_similarity(x1[i], x2[i])
        scores[i] = cal_score(score, target[i].item())
    if reduction == 'mean':
        return scores.mean()
    elif reduction == 'sum':
        return scores.sum()

if __name__ == '__main__':
    A = torch.tensor([[1.0617, 1.3397, -0.2303],
                      [0.3459, -0.9821, 1.2511]])
    B = torch.tensor([[-1.3730, 0.0183, -1.2268],
                      [0.4486, -0.6504, 1.5173]])
    Tar = torch.tensor([1, -1])
    criterion = nn.CosineEmbeddingLoss()
    score = criterion(A, B, Tar)
    score_my = criterion_my(A, B, Tar)
    score_my2 = criterion_my2(A, B, Tar)
    score_mse=criterion_my_mse(A,B,Tar)
    print(score)
    print(score_my)
    print(score_my2)  # tensor(1.1646)
    print(score_mse)

