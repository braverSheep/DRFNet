import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
import os, random

from model import *
from data_input import getloader

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

data_dir = "../../../datasets/SEED/seed_4s/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

y_true = None
y_score = None


def get_model(subj_num=1,session_num=1):
    model = Model2().to(device)
    
    model_weight_path = f"./Pro_log/best_model_subj{subj_num}_session{session_num}.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    return model

def acc_subj(num_subj=1,session_num=1,batch_size = 32):
    global y_score
    global y_true

    net = get_model(subj_num=num_subj, session_num=session_num)
    tes_loader = getloader(data_dir, subj_num=num_subj, session_num=session_num, batch_size=batch_size, is_train=False, is_shuffle=True)
    tes_acc = 0
    with torch.no_grad():
        for step, data in enumerate(tes_loader):
            images, labels = data
            outputs,_ = net(images)

            predict_y = torch.max(outputs, dim=1)[1]
            tes_acc += (predict_y == labels.to(device)).sum().item()

            if y_score is None:
                y_score = predict_y
                y_true = labels
            else:
                y_score = torch.cat((y_score, predict_y), dim=0)
                y_true = torch.cat((y_true, labels), dim=0)

        tes_acc = tes_acc / len(tes_loader.dataset)
        print(tes_acc)
    return tes_acc


def save_predictions(y_true, y_score, session_num):
    # 保存y_true和y_score
    torch.save({'y_true': y_true, 'y_score': y_score}, f'D:\\research\\文章\\multi_region\\SEED_session{session_num}.pth')

if __name__ == '__main__':

    best_Acc = []
    for i in range(1, 16):
        print(f'------------------start subect:{i}--------------------- ')
        best_model_acc = acc_subj(num_subj=i,session_num=1,batch_size = 64)

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
    save_predictions(y_true, y_score, session_num=1)


    #----------------------------------------画ROC曲线

    y_score = y_score.cpu().numpy()
    y_true = y_true.cpu().numpy()
    print(y_score)

    # 1.计算混淆矩阵
    cm = confusion_matrix(y_true, y_score)
    conf_matrix = pd.DataFrame(cm, index=['SA','NE','HA'], columns=['SA','NE','HA'])  #数据有4个类别

    # 转化为百分比
    conf_matrix_percent = conf_matrix.to_numpy() / conf_matrix.sum(axis=1).to_numpy()[:, np.newaxis] * 100

    # 画出混淆矩阵
    fig, ax = plt.subplots(figsize=(7, 6))
    conf_matrix_custom_labels = ['SA','NE','HA']
    sns.heatmap(conf_matrix_percent, annot=True, annot_kws={"size": 14}, cmap="Blues", fmt='.2f',
                xticklabels=conf_matrix_custom_labels, yticklabels=conf_matrix_custom_labels)

    # 设置字体为Times New Roman
    plt.xlabel('Predicted classes(%)', fontsize=14, family='Times New Roman', fontstyle='italic')
    plt.ylabel('True classes(%)', fontsize=14, family='Times New Roman', fontstyle='italic')
    plt.xticks(fontsize=14, family='Times New Roman')
    plt.yticks(fontsize=14, family='Times New Roman')

    # plt.savefig('confusion.pdf', bbox_inches='tight')
    plt.show()
