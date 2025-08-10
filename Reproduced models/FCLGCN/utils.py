import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import math
import torch
import os


class WarmUpStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer, cold_epochs: int, warm_epochs: int, step_size: int,
                 gamma: float = 0.1, last_epoch: int = -1):

        super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)
        self.cold_epochs = cold_epochs
        self.warm_epochs = warm_epochs
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        if self.last_epoch < self.cold_epochs:
            return [base_lr * 0.1 for base_lr in self.base_lrs]
        elif self.last_epoch < self.cold_epochs + self.warm_epochs:
            return [
                base_lr * 0.1 + (1 + self.last_epoch - self.cold_epochs) * 0.9 * base_lr / self.warm_epochs
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * self.gamma ** ((self.last_epoch - self.cold_epochs - self.warm_epochs) // self.step_size)
                for base_lr in self.base_lrs
            ]


class WarmUpExponentialLR(WarmUpStepLR):

    def __init__(self, optimizer: torch.optim.Optimizer, cold_epochs: int, warm_epochs: int,  # lr=lr*gamma
                 gamma: float = 0.1, last_epoch: int = -1):  # last_epoch: int = -1学习率设置为初始值

        self.cold_epochs = cold_epochs
        self.warm_epochs = warm_epochs
        self.step_size = 1
        self.gamma = gamma

        super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

def normalize_A(A, lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()
    return Lnorm


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support


class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, adj):
#         support = torch.matmul(input, self.weight)
#         output = torch.matmul(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'



class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)
