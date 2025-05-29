import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
import os, random
from tqdm import *

from model import *
import CNN

# 设置随机数种子
def seed_torch(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 关闭自动选择最快算法的选项
    torch.backends.cudnn.deterministic = True # 让 CuDNN 以确定性的方式执行
seed_torch()
from data_input import getloader
# from data_input_rotate import getloader

# 参数正则化L2
def parameter_Regular(model,lambada=0.0005):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.norm(param, p=2)
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


data_dir = "../../../datasets/SEED-IV/eeg_feature_smooth/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
exp_dir = cre_prolog()
log_file = f"{exp_dir}/log.txt"


class LSR(nn.Module):
    def __init__(self, n_classes=7, eps=0.1):
        super(LSR, self).__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, outputs, labels):
        # labels.shape: [b,]
        assert outputs.size(0) == labels.size(0)
        n_classes = self.n_classes
        one_hot = F.one_hot(labels, n_classes).float()
        mask = ~(one_hot > 0)
        smooth_labels = torch.masked_fill(one_hot, mask, self.eps / (n_classes - 1))
        smooth_labels = torch.masked_fill(smooth_labels, ~mask, 1 - self.eps)
        ce_loss = torch.sum(-smooth_labels * F.log_softmax(outputs, 1), dim=1).mean() # 标准是用这个
        # ce_loss = torch.sum(-smooth_labels * F.log_softmax(outputs, 1), dim=1)
        # print(ce_loss/ce_loss.mean())
        # ce_loss = F.nll_loss(F.log_softmax(outputs, 1), labels, reduction='mean')
        return ce_loss

class region_loss(nn.Module):
    def __init__(self, beta=0.1):        
        super(region_loss, self).__init__()  
        self.beta = beta

    def forward(self, alpha):
        loss_wt = 0.0
        size = alpha.shape[0]
        # print(size, alpha.shape, alpha[0].shape)
        for i in range(size):
            loss_wt += self.total_loss(alpha[i])
        return  loss_wt/size

    def total_loss(self, alpha, lambda_sim=1.0, lambda_reg=1.0, lambda_imp=1.0, lambda_max_alpha=1.0):
        """
        总损失函数：包括正则化损失、重要性损失、相似度损失和最大化 alpha 值损失

        """
        reg_loss = self.alpha_regularization_loss(alpha)
        imp_loss = self.importance_loss(alpha)
        max_alpha_loss = self.max_alpha_loss(alpha)
        
        # print(reg_loss,imp_loss,max_alpha_loss)
        total = lambda_reg * reg_loss + lambda_imp * imp_loss + lambda_max_alpha * max_alpha_loss
        # print(reg_loss.shape, imp_loss.shape, max_alpha_loss.shape, total.shape)

        return total


    def alpha_regularization_loss(self,alpha):
        """
        alpha 值正则化损失，确保 alpha 值不过于集中或分散
        """
        alpha_mean = alpha.mean(dim=-1, keepdim=True)  # 计算 alpha 的均值
        loss = torch.sum((alpha - alpha_mean) ** 2)  # 计算每个 alpha 与均值的差异
        return loss.unsqueeze(0)

    def importance_loss(self,alpha):
        """
        区域重要性奖励损失，提升 alpha 值大的区域的权重
        """
        # print(alpha)
        # print(torch.log(alpha + 1e-8))
        return -torch.sum(alpha * torch.log(alpha + 1e-8)).unsqueeze(0)  # 加入小的偏移量避免 log(0)

    def max_alpha_loss(self, alpha):
        """
        最大化 alpha 最大值, 并给一个下限
        """
        # print(alpha.shape)
        alphas_part_max = torch.max(alpha, dim=0)[0]
        # print(alphas_part_max)
        return max(torch.Tensor([0]).cuda(), (self.beta - alphas_part_max).unsqueeze(0))



def train_subject(num_subj=1,session_num=1,batch_size = 32,lr = 0.001,epochs = 240,T_max = 40, drop_rate = 0,):

    tra_loader = getloader(data_dir, subj_num=num_subj, session_num=session_num, batch_size=batch_size, is_train=True, is_shuffle=True)
    tes_loader = getloader(data_dir, subj_num=num_subj, session_num=session_num, batch_size=batch_size, is_train=False, is_shuffle=True)
    
    # 定义网络、损失函数、优化器、迭代步骤
    # net = Model1().to(device)
    net = Model2().to(device)
    # net = CNN.cnnBlcok().to(device)
    

    loss_function = nn.CrossEntropyLoss()
    # loss_function = LSR(n_classes=4, eps=0.1)
    loss_function2 = region_loss(beta=0.1)
    optimizer = optim.Adam(net.parameters(), lr=lr)# 0.0001 bs=128, weight_decay=0.0001
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max, eta_min=0, last_epoch=-1) # T_max=20：最大更新步数

    best_acc = 0
    best_epoch = 0
    # # 初始化 tqdm 进度条
    # pbar = tqdm(total=epoch, desc='Training Progress', unit='epoch')
    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        tra_loss = 0
        tes_loss = 0
        tra_acc = 0
        tes_acc = 0
        net.train()
        for step, data in enumerate(tra_loader):
            images, labels = data
            outputs, alphes = net(images)

            optimizer.zero_grad()

            # labels_one_hot = F.one_hot(labels, num_classes=4).float().to(device)
            # loss = F.kl_div(F.log_softmax(outputs, -1), labels_one_hot, reduction="sum")
            loss = loss_function(outputs, labels.to(device)) + loss_function2(alphes) + parameter_Regular(net,lambada=0.01)
            loss.backward()
            optimizer.step()
            
            # 训练集中的损失值和准确
            predict_y = torch.max(outputs, dim=1)[1]
            tra_acc += (predict_y == labels.to(device)).sum().item()
            tra_loss += loss.item()
        # scheduler.step()
        tra_loss = tra_loss / len(tra_loader)
        tra_acc = tra_acc / len(tra_loader.dataset)

        net.eval()  # 在测试过程中关掉dropout方法，不希望在测试过程中使用dropout
        with torch.no_grad():
            for step, data in enumerate(tes_loader):
                images, labels = data
                outputs, alphes = net(images)
                
                loss = loss_function(outputs, labels.to(device)) + loss_function2(alphes) + parameter_Regular(net,lambada=0.01)

                predict_y = torch.max(outputs, dim=1)[1]
                tes_acc += (predict_y == labels.to(device)).sum().item()
                tes_loss += loss.item()

            tes_loss = tes_loss / len(tes_loader)
            tes_acc = tes_acc / len(tes_loader.dataset)
            if tes_acc > best_acc:
                best_acc = tes_acc
                best_epoch = epoch + 1 
                torch.save(net.state_dict(), f"{exp_dir}/best_model_subj{num_subj}_session{session_num}.pth")#_acc{tes_acc:.4f}
            
            # 在进度条上更新当前 epoch 及最好的验证精度

        print(f'\n Epoch {epoch} tra_loss: {tra_loss:.4f}, tra_acc: {tra_acc:.4f}, tes_acc: {tes_acc:.4f}, tes_loss: {tes_loss:.4f}, best_acc: {best_acc:.4f}, best_epoch: {best_epoch} ')
        with open(log_file, 'a') as file:  
            line_to_write = f'Epoch {epoch} tra_loss: {tra_loss:.4f}, tra_acc: {tra_acc:.4f}, tes_acc: {tes_acc:.4f}, tes_loss: {tes_loss:.4f}, best_acc: {best_acc:.4f}, best_epoch: {best_epoch}\n'  
            file.write(line_to_write) 
        if tes_acc>0.9999:
            break

    return best_acc, best_epoch

if __name__ == '__main__':

    # 超参数设置
    batch_size = 64
    lr = 0.0008
    epoch = 180
    T_max = 18
    drop_rate = 0
  
    # 开始训练
    best_Acc = []
    for i in range(1, 16):
        print(f'------------------start subect:{i}--------------------- ')
        with open(log_file, 'a') as file:  
            file.write(f'\n---------------------------------------------------------------------start subect:{i}---------------------------------------------------------------------\n ')
        best_model_acc, best_epoch = train_subject(num_subj=i, session_num=1, batch_size=batch_size, lr=lr, epochs=epoch, T_max=T_max, drop_rate=drop_rate,)

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

    with open(log_file, 'a') as file:  
            file.write(result_string)
    print(result_string)