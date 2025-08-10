import torch.nn as nn
import torch
import numpy as np
from audtorch.metrics.functional import pearsonr
device    = torch.device('cuda')
from statsmodels.tsa.stattools import grangercausalitytests



def Pearsoner_M(data,deta):
    channel_num, feature_num = data.shape 
    for cNum in range(0, channel_num):
        axis = data[cNum,:]
        axis = axis[np.newaxis, :]
        axis = np.repeat(axis, channel_num, axis = 0)
        resultP = pearsonr(torch.from_numpy(data),torch.from_numpy(axis))

        zeros_M = torch.zeros(channel_num,1)
        zeros_M[cNum,:] = 1
        resultP = resultP - zeros_M
        if cNum == 0:
            Adj = resultP
        else:
            Adj = np.concatenate([Adj,resultP],axis=1)
    # Adj[Adj >= deta]     = 1
    # Adj[Adj <  deta]     = 0
    Adj_sort = np.sort(Adj.reshape(channel_num*channel_num))
    # print(channel_num*channel_num - deta, channel_num, deta)
    Adj_ind  = Adj_sort[channel_num*channel_num - deta]
    Adj[Adj >= Adj_ind]     = 1
    Adj[Adj <  Adj_ind]     = 0




    return Adj.astype('float32')



def cal_grangercausality(array1,array2):
    arr=np.zeros((array1.shape[0],array2.shape[0]))
    for i in range(array1.shape[0]):
       for j in range(array2.shape[0]):
           tes=np.vstack((array1[i], array2[j]))
           tes=tes.T
           rels=grangercausalitytests(tes, maxlag=2,verbose=False)
           d1=rels[1][0]['ssr_ftest'][1]
           d2=rels[1][0]['ssr_chi2test'][1]
           d3=rels[1][0]['lrtest'][1]
           d4=rels[1][0]['params_ftest'][1]
           d11 =rels[2][0]['ssr_ftest'][1]
           d12 =rels[2][0]['ssr_chi2test'][1]
           d13 =rels[2][0]['lrtest'][1]
           d14 =rels[2][0]['params_ftest'][1]
           if d1 <0.05 and d2<0.05 and d3<0.05 and d4<0.05 and d11 <0.05 and d12<0.05 and d13<0.05 and d14<0.05:
               d = 1
           else:
               d = 0
           arr[i][j]=d
    arr=arr * (1-np.eye(array1.shape[0]))
    return arr












