import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GraphConvolution, Linear, normalize_A, generate_cheby_adj


class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)

        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result, adj


class DGCNN(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels, num_classes=3):
        # in_channels(int): The feature dimension of each electrode.
        # num_electrodes(int): The number of electrodes.
        # k_adj(int): The number of graph convolutional layers.
        # out_channel(int): The feature dimension of  the graph after GCN.
        # num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        # self.layer1 = GCN(in_channels, out_channels)

        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes * out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
        nn.init.uniform_(self.A, 0.01, 0.5)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # data can also be standardized offline
        L = normalize_A(self.A)
        result, adj = self.layer1(x, L)
        adj = adj[-1]
        # result = result.reshape(x.shape[0], -1)
        # result = self.fc(result)
        return result, adj


if __name__ == "__main__":
    x = torch.randn(16,62,5).cuda()

    net =  DGCNN(5, 62, 2, 16).cuda()

    y,_ = net(x)

    print(y.shape)