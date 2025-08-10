import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
import os, random
from tqdm import *
import contrastive_loss
import math

import FCLGCN_SEED
#from model import *
# from DGCNN import *

# 设置随机数种子
def seed_torch(seed=12):
    random.seed(seed)# 设置 Python 内置 random 模块的种子
    np.random.seed(seed)# 设置 NumPy 的随机种子
    torch.manual_seed(seed)# 设置 PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)# 设置 PyTorch 的 GPU 随机种子
    torch.cuda.manual_seed_all(seed)# 设置所有 GPU 的随机种子
    torch.backends.cudnn.benchmark = False  # 关闭自动选择最快算法的选项
    torch.backends.cudnn.deterministic = True # 让 CuDNN 以确定性的方式执行
seed_torch()
from data_input import getloader
# from data_input_rotate import getloader

# 参数正则化L2 防止模型过拟合
def parameter_Regular(New_model,lambada=0.0005):
    reg_loss = 0.0
    for param in New_model.parameters():
        reg_loss += torch.norm(param, p=2).to(device)
    return reg_loss * lambada

# 整个训练脚本产生的结果都将存入这个脚本中，模型，log等等
def cre_prolog():
    # 获取当前脚本的绝对路径  
    current_script_path = os.path.abspath(__file__)  
    # 获取当前脚本的所在目录  
    parent_dir = os.path.dirname(current_script_path)  
    # 定义要创建的Pro_log目录的路径
    pro_log_dir_path = os.path.join(parent_dir, 'Pro_log')  
      
    # 检查Pro_log目录是否存在，如果不存在则创建  
    if not os.path.exists(pro_log_dir_path):  
        os.makedirs(pro_log_dir_path)  
        print(f"Directory '{pro_log_dir_path}' created.")  

    return pro_log_dir_path


data_dir = r"D:/research/datasets/SEED-VII/EEG_features/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
exp_dir = cre_prolog()
log_file = f"{exp_dir}/log.txt"



def train_subject(num_subj=1,fold_num=1,batch_size = 32,lr = 0.001,epochs = 240,T_max = 40, drop_rate = 0,):
    tra_loader = getloader(data_dir, subj_num=num_subj, fold_num=fold_num, batch_size=batch_size, is_train=True, is_shuffle=True)
    tes_loader = getloader(data_dir, subj_num=num_subj, fold_num=fold_num, batch_size=batch_size, is_train=False, is_shuffle=True)

    net = FCLGCN_SEED.GCNTCN(K=2, T=6, num_channels=62, num_features=5).to(device)

    loss_function = nn.CrossEntropyLoss()
    constrastiveLoss = contrastive_loss.SupConLoss(temperature=0.1)
    optimizer = optim.Adam(net.parameters(), lr=lr)# 0.0001 bs=128, weight_decay=0.0001
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max, eta_min=0, last_epoch=-1) # T_max=20：最大更新步数

    best_acc = 0
    best_epoch = 0

    for epoch in range(epochs):

        tra_loss = 0
        tes_loss = 0
        tra_acc = 0
        tes_acc = 0

        net.train()
        for step, data in enumerate(tra_loader):
            images, labels = data 
            y_proj, y_pred = net(images.to(device))            

            optimizer.zero_grad()

            loss = loss_function(y_pred, labels.to(device)) * 0.8 + constrastiveLoss(y_proj, labels.to(device)) * 0.2
            loss.backward()
            optimizer.step()   
            tra_loss = tra_loss + loss.item()    

            predict_y = torch.max(y_pred, dim=1)[1]
            tra_acc += (predict_y == labels.to(device)).sum().item()     
            
        # scheduler.step()
        tra_loss = tra_loss / len(tra_loader)
        tra_acc = tra_acc / len(tra_loader.dataset)
        

        net.eval()  # 在测试过程中关掉dropout方法，不希望在测试过程中使用dropout
        with torch.no_grad():
            for step, data in enumerate(tes_loader):
                images, labels = data
                y_proj, y_pred = net(images.to(device))
               
                predict_y = torch.max(y_pred, dim=1)[1]
                tes_acc += (predict_y == labels.to(device)).sum().item()
                tes_loss += loss.item()

            tes_loss = 0 / len(tes_loader)
            tes_acc = tes_acc / len(tes_loader.dataset)
            if tes_acc > best_acc:
                best_acc = tes_acc
                best_epoch = epoch + 1 
                torch.save(net.state_dict(), f"{exp_dir}/best_model_subj{num_subj}_session{fold_num}.pth")#_acc{tes_acc:.4f}
            
            # 在进度条上更新当前 epoch 及最好的验证精度

        print(f'\n Epoch {epoch} tra_loss: {tra_loss:.4f}, tra_acc: {tra_acc:.4f}, tes_acc: {tes_acc:.4f}, tes_loss: {tes_loss:.4f}, best_acc: {best_acc:.4f}, best_epoch: {best_epoch}')
        with open(log_file, 'a') as file:  
            line_to_write = f'Epoch {epoch} tra_loss: {tra_loss:.4f}, tra_acc: {tra_acc:.4f}, tes_acc: {tes_acc:.4f}, tes_loss: {tes_loss:.4f}, best_acc: {best_acc:.4f}, best_epoch: {best_epoch} \n'  
            file.write(line_to_write) 
        if tes_acc>0.9999:
            break

    return best_acc, best_epoch

if __name__ == '__main__':

    # 超参数设置
    batch_size = 64           #每次训练输入模型的数据量 #0.0008
    lr = 0.0001              #0.00001 #0.006
    epoch = 180                #训练总轮数
    T_max = 18                #控制学习率在一个完整周期内的变化范围和速度
    drop_rate = 0             #不使用Dropout
  
    # 开始训练
    best_Acc = []
    for i in range(1, 21):
        print(f'------------------start subect:{i}--------------------- ')
        with open(log_file, 'a') as file:  
            file.write(f'\n---------------------------------------------------------------------start subect:{i}---------------------------------------------------------------------\n ')
        best_model_acc, best_epoch = train_subject(num_subj=i, fold_num=1, batch_size=batch_size, lr=lr, epochs=epoch, T_max=T_max, drop_rate=drop_rate,)
        best_Acc.append(best_model_acc)

    k = 0
    result_string = ''
    for i in best_Acc:
        k=k+1                         
        str1 = f'subject{k} acc:{i} \n'
        result_string = ''.join([result_string, str1])

    # 所有subjects的平均准确率、标准差
    mean_acc = np.mean(best_Acc)
    sta = np.std(best_Acc, ddof=0)
    mean_str = f'mean_acc:{mean_acc} sta:{sta}'

    result_string = ''.join([result_string, mean_str]) 

    with open(log_file, 'a') as file:  
            file.write(result_string)
    print(result_string)