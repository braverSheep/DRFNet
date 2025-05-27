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
                y_score = F.softmax(outputs, dim=1)
                y_true = labels
            else:
                y_score = torch.cat((y_score, F.softmax(outputs, dim=1)), dim=0)
                y_true = torch.cat((y_true, labels), dim=0)

        tes_acc = tes_acc / len(tes_loader.dataset)
        print(tes_acc)
    return tes_acc

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


    y_score = y_score.cpu().numpy()
    y_true = y_true.cpu().numpy()

    #----------------------------------------画PR曲线


    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    from sklearn.preprocessing import label_binarize
    from scipy.interpolate import interp1d

    # 假设 y_true 和 y_score 已经定义
    # y_true: 原始标签，形状 (N,)；取值范围是 [0, n_classes - 1]
    # y_score: 预测的概率分数，形状为 (N, n_classes)

    plt.rcParams['font.family'] = 'Times New Roman'

    n_classes = 4
    class_labels = ['NE', 'SA', 'FE', 'HA']
    colors = ['#ffb347', '#aec6cf', '#77dd77', '#cdb4db']

    # 将真实标签二值化
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

    # 1. 绘制每个类别的 PR 曲线并存储 (precision, recall)
    plt.figure(figsize=(5, 4))
    precision_list, recall_list = [], []
    aucs = []

    for i, color in enumerate(colors):
        precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        auc_i = average_precision_score(y_true_bin[:, i], y_score[:, i])
        aucs.append(auc_i)
        
        # 将各类别曲线设为“虚线”（--）
        plt.plot(
            recall_i,
            precision_i,
            color=color,
            linestyle='--',   # <--- 改成虚线
            lw=2,
            label=f'{class_labels[i]} (area = {auc_i:.4f})'
        )
        
        precision_list.append(precision_i)
        recall_list.append(recall_i)



    # 2. 计算并绘制 Macro-average PR 曲线（实线）
    all_recall = np.linspace(0, 1, 100)

    precision1, recall2, _ = precision_recall_curve(y_true_bin.ravel(),y_score.ravel())
    average_precision = average_precision_score(y_true_bin, y_score, average="macro")

    plt.plot(
        recall2,
        precision1,
        color='#8B0000',
        linestyle='-',  # <--- 改成实线
        lw=2,
        label=f'Macro-average (area = {average_precision:.4f})'
    )

    # # 3. 计算并绘制 Micro-average PR 曲线（实线）
    # precision3, recall4, _ = precision_recall_curve(y_true_bin.ravel(),y_score.ravel())
    # average_precision2 = average_precision_score(y_true_bin, y_score, average="micro")

    # plt.plot(
    #     recall4,
    #     precision3,
    #     color='#4b0082',
    #     linestyle='-',  # <--- 改成实线
    #     lw=2,
    #     label=f'Micro-average (area = {average_precision2:.4f})'
    # )


    print('Average precision score, macro-averaged over all classes: {0:0.4f}'.format(average_precision))

    # 4. 图形样式设置并显示
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    # plt.title('PR Curve: SEED-IV', fontsize=16)
    plt.legend(loc='best', fontsize=14)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    # plt.savefig(r'D:\research\文章\multi_region\new paper\pr_seeds_IV.png', dpi=300)
    plt.show()


