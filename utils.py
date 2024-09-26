import torch
import torch.nn.functional as F

def criterion(input, target, size_average=True):
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l