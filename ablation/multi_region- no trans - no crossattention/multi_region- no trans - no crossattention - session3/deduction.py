import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
import os, random

from model import *
from model2 import model_cnn
from model_all import modelCNN_Tran
from model4 import resmodel_cnn
import model_DWConv
import model_tran
from data_input import getloader

data_dir = "../../datasets/SEED-IV/eeg_feature_smooth/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_model(subj_num=1,session_num=1):
    model = model_cnn(img_size=112, in_channels=5, num_class=4)
    
    model_weight_path = f"./Pro_log/best_model_subj{subj_num}_session{session_num}.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    return model

def acc_subj(num_subj=1,session_num=1,batch_size = 32):
    net = get_model(subj_num=num_subj, session_num=session_num)

    tes_loader = getloader(data_dir, subj_num=num_subj, session_num=session_num, batch_size=batch_size, is_train=False, is_shuffle=False)
    tes_acc = 0
    with torch.no_grad():
        for step, data in enumerate(tes_loader):
            images, labels = data
            outputs = net(images)

            predict_y = torch.max(outputs, dim=1)[1]
            tes_acc += (predict_y == labels).sum().item()

        tes_acc = tes_acc / len(tes_loader.dataset)
        print(tes_acc)
    return tes_acc

if __name__ == '__main__':

    best_Acc = []
    for i in range(1, 16):
        print(f'------------------start subect:{i}--------------------- ')
        best_model_acc = acc_subj(num_subj=i,session_num=1,batch_size = 32)

        best_Acc.append(best_model_acc)

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

