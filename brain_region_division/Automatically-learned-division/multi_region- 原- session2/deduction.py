import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
import os, random

from model import *

from data_input import getloader

data_dir = "../../../datasets/SEED-IV/eeg_feature_smooth/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_f1(y_true, y_pred, average='macro'):
    from sklearn.metrics import precision_score, recall_score, f1_score
    """
    计算并返回多分类F1分数，给定真实标签和预测标签（支持PyTorch张量输入）
    
    参数:
    y_true (Tensor): 真实标签 (PyTorch张量)
    y_pred (Tensor): 预测标签 (PyTorch张量)
    average (str): 多类别平均方式 ('binary', 'macro', 'micro', 'weighted', 'samples')

    返回:
    float: F1分数
    """
    # 如果张量在GPU上，先移到CPU
    if y_true.is_cuda:
        y_true = y_true.cpu()
    if y_pred.is_cuda:
        y_pred = y_pred.cpu()
    
    # 转换为NumPy数组
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    # 计算并返回F1分数
    return f1_score(y_true, y_pred, average=average)

def get_model(subj_num=1,session_num=1):
    model = Model2().to(device)
    
    model_weight_path = f"./Pro_log/best_model_subj{subj_num}_session{session_num}.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    return model

def acc_subj(num_subj=1,session_num=1,batch_size = 32):
    net = get_model(subj_num=num_subj, session_num=session_num)

    tes_loader = getloader(data_dir, subj_num=num_subj, session_num=session_num, batch_size=batch_size, is_train=False, is_shuffle=False)
    tes_acc = 0
    count = 0
    with torch.no_grad():
        for step, data in enumerate(tes_loader):
            images, labels = data
            outputs,_  = net(images)

            predict_y = torch.max(outputs, dim=1)[1]
            if count == 0:
                y_pred = predict_y
                y_true = labels.to(device)
            else:
                y_pred = torch.cat((y_pred, predict_y), dim=0)
                y_true = torch.cat((y_true, labels.to(device)), dim=0)
            tes_acc += (predict_y == labels.to(device)).sum().item()
            count = count+1

        tes_acc = tes_acc / len(tes_loader.dataset)
        print(tes_acc)
        f1 = calculate_f1(y_true,y_pred)
        print("f1", f1)
    return tes_acc,f1

if __name__ == '__main__':

    best_Acc = []
    F1 = []
    for i in range(1, 16):
        print(f'------------------start subect:{i}--------------------- ')
        best_model_acc,f1 = acc_subj(num_subj=i,session_num=2,batch_size = 64)

        best_Acc.append(best_model_acc)
        F1.append(f1)

    k = 0
    result_string = ''
    for i in best_Acc:
        k=k+1
        str1 = f'subject{k} acc:{i} \n'
        result_string = ''.join([result_string, str1])

    # 所有subjects的平均准确率、标准差
    mean_acc = np.mean(best_Acc)
    sta = np.std(best_Acc, ddof=1)
    mean_str = f'mean_acc:{mean_acc} sta:{sta}'

    result_string = ''.join([result_string, mean_str]) 

    print(result_string)

    print("f1_aver:", np.mean(F1), np.std(F1, ddof=1))




