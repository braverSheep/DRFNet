import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import Module


class GraphConvolution(nn.Module):

    def __init__(self, num_in, num_out, bias=False):

        super(GraphConvolution, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out).cuda())
        nn.init.kaiming_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_out).cuda())
            nn.init.constant_(self.bias,0.0)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight) 
        if self.bias is not None:
            return out + self.bias
        else:
            return out
