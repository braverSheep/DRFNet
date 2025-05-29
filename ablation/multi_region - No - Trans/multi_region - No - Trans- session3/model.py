import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from CNN import cnnBlcok
from Transformer import Transformer_encoder, self_attention
from DGCNN import GCN
from positional_encodings.torch_encodings import PositionalEncoding1D

class ScaledDotProductAttention(nn.Module):
    def __init__(self,n_head,d_k,dropout=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context
    
class CrossAttention(nn.Module):#n_layers=2,n_heads=5,d_model=self.band_num*2,d_k=8,d_v=8,d_ff=10
    def __init__(self,n_heads,d_model,d_k,d_v,dropout=0.):
        super(CrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.n_heads,self.d_k,dropout)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_V.weight)
        nn.init.xavier_normal_(self.fc.weight)
     
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        output = self.dropout(output)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)

class cnn_tran_fuse(nn.Module):
	def __init__(self, in_channels=5):
		super(cnn_tran_fuse, self).__init__()
		# self.downCh= nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
		self.atten = CrossAttention(n_heads=5, d_model=45, d_k=9, d_v=9, dropout=0.)

	def forward(self, cnn, tran):
		b,c = cnn.shape[0], cnn.shape[1]
		# print(cnn.shape)
		cnn = cnn.view(b, c, -1).permute(0,2,1)
		tran= tran.view(b, c, -1).permute(0,2,1)
		x = self.atten(tran, cnn, cnn)
		x = x.view(x.shape[0],-1)

		return x


class region_att(nn.Module):
	def __init__(self, inchannel=512):
		super(region_att, self).__init__()
		self.linear = nn.Linear(inchannel,1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		alph = self.linear(x)
		alph = self.sigmoid(alph)

		return alph


class Model2(nn.Module):
	def __init__(self, in_channels=5, num_class=4):
		super(Model2, self).__init__()
		# 参数设置
		self.branches = 7
		self.out_channels = 45
		self.dim = 45
		self.num_heads = 5
		self.fuse_dim = 90

		# 主要结构
		# 局部
		self.cnnblock = torch.nn.ModuleList([cnnBlcok(in_channels=in_channels, out_channels=self.out_channels, is_change_scale=True) for _ in range(self.branches)])
		self.linear = torch.nn.ModuleList([nn.Linear(in_channels, self.dim) for _ in range(self.branches + 1)]) # transformer前改变数据维度
		# self.tranblock = torch.nn.ModuleList([Transformer_encoder(dim=self.dim, num_heads=self.num_heads,tran_num=3) for _ in range(self.branches)])
		self.cnn_tran_fuse = torch.nn.ModuleList([cnn_tran_fuse(in_channels=self.out_channels) for _ in range(self.branches)])

		self.potions = torch.nn.ModuleList([PositionalEncoding1D(self.dim),
										   PositionalEncoding1D(self.dim),
										   PositionalEncoding1D(self.dim),
										   PositionalEncoding1D(self.dim),
										   PositionalEncoding1D(self.dim),
										   PositionalEncoding1D(self.dim),
										   PositionalEncoding1D(self.dim),
			])
		self.linear2 = torch.nn.ModuleList([ 
			nn.Sequential(nn.Linear((self.out_channels*9), self.fuse_dim),nn.BatchNorm1d(self.fuse_dim)),
			nn.Sequential(nn.Linear((self.out_channels*9), self.fuse_dim),nn.BatchNorm1d(self.fuse_dim)),
			nn.Sequential(nn.Linear((self.out_channels*9), self.fuse_dim),nn.BatchNorm1d(self.fuse_dim)),
			nn.Sequential(nn.Linear((self.out_channels*9), self.fuse_dim),nn.BatchNorm1d(self.fuse_dim)),
			nn.Sequential(nn.Linear((self.out_channels*9), self.fuse_dim),nn.BatchNorm1d(self.fuse_dim)),
			nn.Sequential(nn.Linear((self.out_channels*25), self.fuse_dim),nn.BatchNorm1d(self.fuse_dim)),
			nn.Sequential(nn.Linear((self.out_channels*9), self.fuse_dim),nn.BatchNorm1d(self.fuse_dim)),]
		)


		# first region attention
		self.region_alphes = torch.nn.ModuleList([
							region_att(self.fuse_dim),
							region_att(self.fuse_dim),
							region_att(self.fuse_dim),
							region_att(self.fuse_dim),
							region_att(self.fuse_dim),
							region_att(self.fuse_dim),
							region_att(self.fuse_dim),
			])
		# second region attion
		self.region_beta = torch.nn.ModuleList([
							region_att(self.fuse_dim+self.fuse_dim),
							region_att(self.fuse_dim+self.fuse_dim),
							region_att(self.fuse_dim+self.fuse_dim),
							region_att(self.fuse_dim+self.fuse_dim),
							region_att(self.fuse_dim+self.fuse_dim),
							region_att(self.fuse_dim+self.fuse_dim),
							region_att(self.fuse_dim+self.fuse_dim),
			])


		self.fc = nn.Linear(self.fuse_dim*2, num_class)
		

	def forward(self, data):
		keys = list(data.keys())
		# print(keys)
		branches = []

		# 七个局部
		for i in range(self.branches):
			
			cnnx = self.cnnblock[i](data[keys[i]].cuda())
			# print(cnnx.shape)
			linear = self.linear[i](data[keys[i + self.branches]].cuda())
			# print(linear.shape)
			# tranx= self.tranblock[i](linear + self.potions[i](linear))
			# print(tranx.shape)

			# sqrt_n = int(tranx.shape[1]**0.5)
			# tranx = tranx.view(tranx.shape[0], sqrt_n, sqrt_n, -1).permute(0, 3, 1, 2) # 将Transformer的输出转换成[b,c,w,h]
			x = self.cnn_tran_fuse[i](cnnx, cnnx)
			# print(x.shape)
			branches.append(self.linear2[i](x)) # branches.append(self.linear2[i](x).unsqueeze(dim=1))
			

		# 第一个region attention
		for i in range(self.branches):
			alph = self.region_alphes[i](branches[i])
			# print(branches[i].shape)
			# print(alph.shape) 
			if i==0:
				alphes = alph
			else:
				alphes = torch.cat((alphes,alph), dim=1)

		
		# 对alpha进行归一化
		alphes_nor = F.softmax(alphes, dim=1)
		# print(alphes_nor.shape)
		# print(branches[0].shape)

		# alphes_nor[:,0] 的维度是[b]，无法广播； alphes_nor[:,0:1]的维度[b,1]才可以用以广播
		Fm = branches[0] * alphes_nor[:,0:1] + branches[1] * alphes_nor[:,1:2] + branches[2] * alphes_nor[:,2:3] + branches[3] * alphes_nor[:,3:4] + branches[4] * alphes_nor[:,4:5] + branches[5] * alphes_nor[:,5:6] + branches[6] * alphes_nor[:,6:7] 
		
		# 第二个region attention
		for i in range(self.branches):
			branches[i] = torch.cat((branches[i], Fm), dim=1)
			beta = self.region_beta[i](branches[i])
			if i==0:
				betas = beta
			else:
				betas = torch.cat((betas,beta), dim=1)

		# 对beta*beta进行归一化
		# beta_nor = torch.mul(betas, alphes)
		# beta_nor = F.softmax(beta_nor, dim=1)
		beta_nor = F.softmax(betas, dim=1)
		Fm = branches[0] * beta_nor[:,0:1] + branches[1] * beta_nor[:,1:2] + branches[2] * beta_nor[:,2:3] + branches[3] * beta_nor[:,3:4] + branches[4] * beta_nor[:,4:5] + branches[5] * beta_nor[:,5:6] + branches[6] * beta_nor[:,6:7] 

		x = self.fc(Fm)#.view(x.shape[0], -1)
		# y = torch.max(alphes, dim=0)[0]
		# print(y.shape)
		# print(alphes.shape)
		return x, alphes# alphes 返回region attention，最大的alpha用作损失

if __name__=='__main__':

    net = Model2().cuda()
    from data_input import getloader
    SEEDIV_data_dir = "../../../datasets/SEED-IV/eeg_feature_smooth/"

    data_loader = getloader(SEEDIV_data_dir, subj_num=1, session_num=1, batch_size=32, is_train=True, is_drop_last=False, is_shuffle=True)
    
    for step, data in enumerate(data_loader, start=0):
        data, labels = data
        y,  _ = net(data)
        print(y.shape)
        break
    
    
    