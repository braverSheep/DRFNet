import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from utils import normalize_A, generate_cheby_adj, generate_R
import math
import numpy as np


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        
        device = input_list[0].device
        self.random_matrix = [matrix.to(device) for matrix in self.random_matrix]  

        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=100.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


class GNN1(nn.Module):
    def __init__(self, xdim, K, num_out, dropout, adapt=False):
        super(GNN1, self).__init__()
        self.k_adj = K
        self.gc1   = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2],num_out))
        self.adapt = adapt
        self.BN = nn.BatchNorm1d(num_out)
        if self.adapt:
            self.K = nn.Parameter(torch.FloatTensor(xdim[2],64).cuda())
            nn.init.kaiming_normal_(self.K)
            self.Q = nn.Parameter(torch.FloatTensor(xdim[2],64).cuda())
            nn.init.kaiming_normal_(self.Q)
    def forward(self,x,L):
        if self.adapt:
            K = torch.matmul(x, self.K)
            Q = torch.matmul(x, self.Q)
            M = torch.matmul(K, Q.transpose(1, 2))
            M = torch.sigmoid(M)
            L = L * M
        else:
            M=L
        adj = generate_cheby_adj(L,self.k_adj)
        for i in range(len(self.gc1)):
            if i == 0:
                result1 = self.gc1[i](x,adj[i])
            else:
                result1 += self.gc1[i](x,adj[i])
        result = F.relu(result1)
        return result, L, M

class GNN(nn.Module):
    def __init__(self,xdim, k_adj, num_out, dropout, nclass, domain_adaptation=False):
        super(GNN, self).__init__()
        self.layer1 = GNN1(xdim, k_adj, num_out, dropout, adapt=True)
        self.BN1 = nn.BatchNorm1d(xdim[-1])

        self.BN2 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(xdim[1] * num_out*1, 512)
        self.fc2 = nn.Linear(512, nclass)

    def forward(self,x,adj_pa,alpha=0):

        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)

        A_all = adj_pa
        L_all = normalize_A(A_all, symmetry=False)
        result, L_all1, M_all = self.layer1(x, L_all)
        # (B,N,F)->(B,N,Out)
        # result:[batch_size,node,num_out]

        result1 = result.reshape(x.shape[0], -1)
        # result:[batch_size,node*num_out]



        result = F.relu(self.BN2(self.fc1(result1)))
        # result:[batch_size,node*num_out]->[batch_size,512] 类似图融合了
        result_output = self.fc2(result)
        return result_output, result, [A_all, L_all1, M_all]



if __name__=="__main__":
    x = torch.randn(16, 62, 5).cuda()
    adj = torch.randn(62,62).cuda()
    model = GNN([16,62,5], 2, 32, 0.5, 3).cuda()

    y,_,_ = model(x, adj, 0)

    print(y.shape)