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


from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def plot_roc(y_test, y_scores, classes=None):
    """
    绘制多类分类的ROC曲线并计算宏平均AUC。

    参数：
    - y_test: 真实标签（必须是整数标签，如0、1、2等）
    - y_scores: 预测的概率，shape为 (n_samples, n_classes)
    - classes: 类别标签的列表，用于确定每一类的标签（可选）
    """

    # 如果没有提供类别标签，使用y_test中的唯一类别
    if classes is None:
        classes = np.unique(y_test)

    # 将y_test转换成one-hot形式
    ytest_one_rf = label_binarize(y_test, classes=classes)

    # 初始化存储AUC, FPR, TPR的字典
    rf_AUC = {}
    rf_FPR = {}
    rf_TPR = {}

    # 对每一个类别计算AUC和FPR, TPR
    for i in range(ytest_one_rf.shape[1]):
        rf_FPR[i], rf_TPR[i], thresholds = roc_curve(ytest_one_rf[:, i], y_scores[:, i])
        rf_AUC[i] = auc(rf_FPR[i], rf_TPR[i])
    print(f"Individual AUC for each class: {rf_AUC}")

    # 合并所有的FPR并排序去重（宏平均法）
    # 我们将所有的FPR合并在一起，并进行去重
    rf_FPR_final = np.unique(np.concatenate([rf_FPR[i] for i in range(ytest_one_rf.shape[1])]))

    # 计算宏平均TPR（True Positive Rate）
    rf_TPR_all = np.zeros_like(rf_FPR_final)
    for i in range(ytest_one_rf.shape[1]):
        rf_TPR_all += np.interp(rf_FPR_final, rf_FPR[i], rf_TPR[i])
    rf_TPR_final = rf_TPR_all / ytest_one_rf.shape[1]

    # 计算最终的宏平均AUC
    rf_AUC_final = auc(rf_FPR_final, rf_TPR_final)
    print(f"Macro Average AUC: {rf_AUC_final}")

    # 绘制所有类别的ROC曲线和宏平均ROC曲线
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(5, 4))#, dpi=300

    # 绘制每个类别的ROC曲线
    
    plt.plot(rf_FPR[0], rf_TPR[0], label=f'NE (area={rf_AUC[0]:.4f})', lw=2, linestyle='--',)
    plt.plot(rf_FPR[1], rf_TPR[1], label=f'SA (area={rf_AUC[1]:.4f})', lw=2, linestyle='--',)
    plt.plot(rf_FPR[2], rf_TPR[2], label=f'FE (area={rf_AUC[2]:.4f})', lw=2, linestyle='--',)
    plt.plot(rf_FPR[3], rf_TPR[3], label=f'HA (area={rf_AUC[3]:.4f})', lw=2, linestyle='--',)

    # 绘制宏平均ROC曲线
    plt.plot(rf_FPR_final, rf_TPR_final, color='#000000', linestyle='-',
             label=f'Macro-average (area={rf_AUC_final:.4f})', lw=3)

    # 绘制45度参考线
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='45 Degree Reference Line')

    # 设置图表的标签和标题
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    # plt.title('Random Forest Classification ROC Curves and AUC', fontsize=14)

    # 显示网格
    plt.grid(linestyle='--', alpha=0.5)

    # 显示图例
    plt.legend(loc='best', fontsize=14)

    # 保存图像
    plt.savefig(r'D:\research\文章\multi_region\new paper\roc_seeds_IV.png', dpi=300)
    # plt.savefig('ROC_Curves_optimized.pdf', format='pdf', bbox_inches='tight')

    # 显示图像
    plt.show()


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
        best_model_acc = acc_subj(num_subj=i,session_num=3,batch_size = 64)

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

    #----------------------------------------画ROC曲线
    class_counts = np.bincount(y_true)
    # 打印每个类别的样本数
    for class_id, count in enumerate(class_counts):
        print(f"Class {class_id}: {count} samples")  
        
    plot_roc(y_true, y_score)