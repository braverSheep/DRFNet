import torch
from torch import nn
import torch.nn.functional as F
# from torch_geometric.data import Batch
from DGCN import DGCNN
# from torch_geometric.nn.norm import BatchNorm

from attention import MultiHeadAttention

class GCNTCN(nn.Module):

    def __init__(self, K, T, num_channels, num_features):
        """
        :param K: 切比雪夫阶数
        :param num_channels: 脑电数据通道数
        :param num_features: 特征数
        """
        super(GCNTCN, self).__init__()
        self.K = K
        self.T = T
        self.num_channels = num_channels
        self.num_features = num_features

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(T):
            self.convs.append(DGCNN(self.num_features, num_channels, K, self.num_features))
            self.batch_norms.append(nn.BatchNorm1d(self.num_features))

        self.sigmoid1 = nn.Sigmoid()
        self.attention = MultiHeadAttention(num_channels * self.num_features, self.num_features)
        self.gru = nn.GRU(num_channels * self.num_features, num_channels, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(self.T)
        self.sigmoid2 = nn.Sigmoid()
        self.linear = nn.Linear(num_channels * self.T, 128)
        self.sigmoid3 = nn.Sigmoid()
        self.projection = nn.Linear(128, 7)
        self.classifier = nn.Linear(num_channels * self.T, 7)

        for param in self.parameters():
            if len(param.shape) < 2:
                nn.init.xavier_uniform_(param.unsqueeze(0))
            else:
                nn.init.xavier_uniform_(param)


    def forward(self, x):
        """
        前向传播
        :param x: list of Batch objects
        [B, T, C, F]
        """
        B, T, C, Dim = x.shape

        # DGCNN layer
        y_list = []
        adj_list = []
        for i in range(self.T):
            feat = x[:, i, :, :] 
            yi, adj = self.convs[i](feat)
            yi = yi.view(-1, Dim)                  # (B*C, Dim)
            yi = self.batch_norms[i](yi)
            y_list.append(yi)
            adj_list.append(adj)

        adj_feature = torch.stack(adj_list)
        y = torch.stack(y_list) # (T, B*C, Dim)
        y = self.sigmoid1(y)
        # GRU layer
        yt = y.transpose(0, 1)
        y = torch.reshape(yt, (B, self.T, -1))
        # import pdb; pdb.set_trace()
        y, attention = self.attention(y,y,y)
        y, hiden_state = self.gru(y)
        y_gru_out = self.batch_norm1(y)
        y_gru_out = self.sigmoid2(y_gru_out)

        out = torch.reshape(y_gru_out, (B, -1))
        y = self.linear(out)

        y = F.normalize(y, dim=0)
        y = self.sigmoid3(y)

        y_proj = self.projection(y)
        y_proj = F.normalize(y_proj, dim=0)
        y_proj = y_proj.unsqueeze(1)

        y_p = self.classifier(out)
        y_pred = self.sigmoid3(y_p)
        return  y_proj, y_pred


if __name__ == "__main__":
    x = torch.randn(16, 6, 62, 5).cuda()

    net = GCNTCN(K=2, T=6, num_channels=62, num_features=5).cuda()

    _, y =  net(x)

    print(y.shape)