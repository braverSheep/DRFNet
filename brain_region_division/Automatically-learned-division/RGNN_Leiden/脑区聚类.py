# aggregate_edge_weight.py

import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #随机梯度下降（SGD）、Adam、Adagrad、RMSprop
import numpy as np
import torch.optim as optim
import os, random
import torch_geometric
import pandas as pd
from scipy.spatial import distance_matrix
import scipy.io
from torch.utils.data import Dataset, DataLoader

data_dir = "D:/dataset/SEED-IV/eeg_feature_smooth/"
root_dir = "D:/dataset/SEED-IV/"

device = torch.device("cpu")
y_true = None
y_score = None


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_data = self.data[index]
        sample_label = self.labels[index]
        return sample_data, sample_label
 
root_dir = "D:/dataset/SEED-IV/"
def get_adjacency_matrix():#在location中找到对应的Channel位置
    channel_order =pd.read_excel(root_dir+"Channel Order.xlsx", header=None)
    channel_location = pd.read_csv(root_dir+"channel_locations.txt", sep="\t",header=None)
    channel_location.columns = ["Channel", "X", "Y", "Z"]
    channel_location["Channel"] = channel_location["Channel"].apply(lambda x: x.strip().upper())
    filtered_df = pd.DataFrame(columns=["Channel", "X", "Y", "Z"])
    for channel in channel_location["Channel"]:
        for used in channel_order[0]:
            if channel == used:
                filtered_df = pd.concat([channel_location.loc[channel_location['Channel']==channel],filtered_df], ignore_index=True)
                # Concatenate each row from the "channel_location" dataframe whose channel name is present in the Excel sheet

    filtered_df = filtered_df.reindex(index=filtered_df.index[::-1]).reset_index(drop=True)#将DataFrame对象filtered_df的索引进行倒序排列，并通过设置drop=True参数来删除原来的索引列。最后将结果赋值给filtered_df变量。
    filtered_matrix = np.asarray(filtered_df.values[:, 1:4], dtype=float)#用于将输入的数据转换为 NumPy 数组

    distances_matrix = distance_matrix(filtered_matrix, filtered_matrix)## Compute a matrix of distances for each combination of channels in the filtered_matrix dataframe
    
    delta = 10
    adjacency_matrix = np.minimum(np.ones([62,62]), delta/(distances_matrix**2)) # Computes the adjacency matrix cells as min(1, delta/d_ij), N.B. zero division error can arise here for the diagonal cells, "1" value will be chosen automatically instead

    return torch.tensor(adjacency_matrix).float()

#-----------Subject dependent----------
def dependent_loaders(batch_size, K):#提取第K个受试者的数据//实验没有考虑时间序列
    seesion1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    seesion2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    seesion3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    labels = [seesion1_label, seesion2_label, seesion3_label]

    trials_seesion1 = [[3,5,6,8,20,22],[0,7,9,11,12,13],[1,4,10,14,16,17],[2,15,18,19,21,23]]#0,1,2,3标签位置
    trials_seesion2 = [[3,4,6,13,18,20],[1,14,15,17,21,23],[0,5,7,10,12,16],[2,8,9,11,19,22]]
    trials_seesion3 = [[11,15,18,19,21,23],[0,3,7,8,10,22],[1,2,9,12,16,20],[4,5,6,13,14,17]]
    trials = [trials_seesion1, trials_seesion2, trials_seesion3]
    eeg_dict = {"train":[], "test":[]}
    eeg = []
    print("--- Subject dependent loader ---")
    for session in range(1,4):
        print("we are in session:{:d}".format(session))
        session_labels = labels[session-1]
        trial = trials[session-1]
        session_folder = "D:/dataset/SEED-IV/eeg_feature_smooth/"+str(session)
        file = ''
        for File in os.listdir(session_folder):#提取第K个受试者的数据
            p=int(File.split("_")[0])
            if p == K:
                file = File
                break
        
        file_path = "{}/{}".format(session_folder, file)
        num = []
        if session==3:
            for i in range(4):#随机取某个同类实验
                num.append(trial[i][4])
                num.append(trial[i][5])
        else :pass
        print(num)
        
        for videoclip in range(24):
            X_psd = scipy.io.loadmat(file_path)['psd_LDS{}'.format(videoclip + 1)]
            X_de = scipy.io.loadmat(file_path)['de_LDS{}'.format(videoclip + 1)]

            y = session_labels[videoclip]
            y = torch.tensor([y]).long()
            edge_index = torch.tensor(np.array([np.arange(62*62)//62, np.arange(62*62)%62]))
            for t in range(X_psd.shape[1]):#每一列？
                x_psd = torch.tensor(X_psd[:, t, :]).float()
                x_psd = (x_psd-x_psd.mean())/x_psd.std()#这里就进行了标准化
                x_de = torch.tensor(X_de[:, t, :]).float()
                x_de = (x_de-x_de.mean())/x_de.std()
                '''
                if videoclip >= 16:#24个实验，前16个为训练集，后8个为测试集
                    eeg_dict["test"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
                else :
                    eeg_dict["train"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
                '''
                #保证测试集和验证集的分类的分布相同
                if session == 3:
                #保证测试集和验证集的分类的分布相同
                   if videoclip == num[0] or videoclip == num[1] or videoclip == num[2] or videoclip == num[3] or videoclip == num[4] or videoclip == num[5] or videoclip == num[6] or videoclip == num[7]:#24个实验，四类情绪分别随机取2个实验，一个session测试集为8个
                       eeg_dict["test"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
                   else:
                       eeg_dict["train"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
                else:  eeg_dict["train"].append((torch_geometric.data.Data(x=x_psd, y=y, edge_index=edge_index),torch_geometric.data.Data(x=x_de, y=y, edge_index=edge_index)))
    loaders_dict = {"train":[], "test":[]}
    
    loaders_dict["train"] = torch_geometric.loader.DataLoader(eeg_dict["train"], batch_size=batch_size, shuffle=True, drop_last=False)#drop_last：如果为True，最后一个不完整的批次将被丢弃
    loaders_dict["test"] = torch_geometric.loader.DataLoader(eeg_dict["test"], batch_size=batch_size, shuffle=True, drop_last=False)            
    
    print("------------------------------")
    return  loaders_dict["test"]


import RGNN
def get_model(subj_num=1,session_num=1):
    adj = get_adjacency_matrix()
    model = RGNN.SymSimGCNNet(62, True, adj, 5, [16], 4, 2, dropout=0.7).to(device)
 
    model_weight_path = f"./emotion_model - 副本/RGNN_{subj_num}.pt"
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    return model

def acc_subj(num_subj=1,session_num=3,batch_size = 16):
    global y_score
    global y_true


    net = get_model(subj_num=num_subj, session_num=session_num)

    tes_loader = dependent_loaders(batch_size, num_subj)
    print(len(tes_loader))
    tes_acc = 0
    with torch.no_grad():
        for batch_train in tes_loader:
            outputs,_ = net(batch_train[1])
            y_list = batch_train[1].y.to('cpu')

            if y_score is None:
                y_score = F.softmax(outputs, dim=1)
                y_true = y_list
            else:
                y_score = torch.cat((y_score, F.softmax(outputs, dim=1)), dim=0)
                y_true = torch.cat((y_true, y_list), dim=0)

            predict_y = torch.max(outputs, dim=1)[1]
            tes_acc += (predict_y == y_list).sum().item() / batch_train[1].y.size(0)

        tes_acc = tes_acc / len(tes_loader)
        print(tes_acc)
    return tes_acc



def aggregate_edge_weights(num_subjects=15, session_num=3, batch_size=16, save_path="aggregated_adj.pt"):
    device = torch.device("cpu")
    
    acc_list = []
    weight_list = []
    result_string = ''

    # 1) 对每个受试者，加载模型、计算准确率并提取 edge_weight
    for subj in range(1, num_subjects + 1):
        print(f"[Subject {subj}] loading model and computing accuracy…")
        model = get_model(subj_num=subj, session_num=session_num).to(device)
        model.eval()

        # 计算该模型的测试准确率
        acc = acc_subj(num_subj=subj, session_num=session_num, batch_size=batch_size)
        acc_list.append(acc)

        # 提取它的 edge_weight 参数（下三角向量形式）
        # 注意：SymSimGCNNet.edge_weight 存储的是下三角（包含对角线）的权重
        w = model.edge_weight.detach().cpu()
        weight_list.append(w)


    # 打印所有subjects的平均准确率、标准差
    mean_acc = np.mean(acc_list)
    sta = np.std(acc_list, ddof=1)
    mean_str = f'mean_acc:{mean_acc} sta:{sta}'

    result_string = ''.join([result_string, mean_str]) 

    print(result_string)


    # 2) 将准确率归一化作为权重
    acc_arr = np.array(acc_list, dtype=np.float32)
    norm_acc = acc_arr / acc_arr.sum()

    # 3) 叠加所有模型的 edge_weight 向量，加权求和
    weights_stack = torch.stack(weight_list, dim=0)             # [15, num_edges]
    norm_acc_tensor = torch.from_numpy(norm_acc).unsqueeze(1)   # [15, 1]
    weighted_lower = (weights_stack * norm_acc_tensor).sum(dim=0)  # [num_edges]

    # 4) 将下三角向量重建为对称的完整邻接矩阵
    # 这里用最后一个 model 的 xs, ys 索引来拼回矩阵
    num_nodes = model.num_nodes
    xs = model.xs.cpu()
    ys = model.ys.cpu()

    agg_adj = torch.zeros((num_nodes, num_nodes), dtype=weighted_lower.dtype)
    agg_adj[xs, ys] = weighted_lower
    agg_adj = agg_adj + agg_adj.T - torch.diag(agg_adj.diagonal())

    # 5) 存盘
    torch.save(agg_adj, save_path)
    print(f"Aggregated adjacency matrix saved to {save_path}")
    return agg_adj



if __name__ == "__main__":
    aggregate_edge_weights()
