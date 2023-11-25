from typing import List
import torch
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, top_k=(1,)) -> List[float]:
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def count_params(model: nn.Module, only_requires_grad: bool = False):
    "count number trainable parameters in a pytorch model"
    if only_requires_grad:
        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    return total_params


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y
    
class WeightedMeter:
    def __init__(self, name: str = None):
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0
        self.val = 0.0

    def update(self, val: float, num: int = 1):
        self.count += num
        self.sum += val * num
        self.avg = self.sum / self.count
        self.val = val

    def reset(self, total: float = 0, count: int = 0):
        self.count = count
        self.sum = total
        self.avg = total / max(count, 1)
        self.val = total / max(count, 1)


class AverageMeter:
    def __init__(self, length: int, name: str = None):
        assert length > 0
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.current: int = -1
        self.history: List[float] = [None] * length

    @property
    def val(self) -> float:
        return self.history[self.current]

    @property
    def avg(self) -> float:
        return self.sum / self.count

    def update(self, val: float):
        self.current = (self.current + 1) % len(self.history)
        self.sum += val

        old = self.history[self.current]
        if old is None:
            self.count += 1
        else:
            self.sum -= old
        self.history[self.current] = val


class TotalMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float):
        self.sum += val
        self.count += 1

    def update_with_weight(self, val: float, count: int):
        self.sum += val*count
        self.count += count

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        if self.count == 0:
            return -1
        return self.sum / self.count


def inner_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss


def intra_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0