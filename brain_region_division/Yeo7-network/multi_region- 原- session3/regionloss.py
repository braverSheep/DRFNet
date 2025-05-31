import torch
import torch.nn.functional as F
import torch.nn as nn


class region_loss(nn.Module):
    def __init__(self, beta=0.1):        
        super(region_loss, self).__init__()  
        self.beta = beta

    def forward(self, alpha):
        loss_wt = 0.0
        size = alpha.shape[0]
        print(size, alpha.shape, alpha[0].shape)
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
        return total


    def alpha_regularization_loss(self,alpha):
        """
        alpha 值正则化损失，确保 alpha 值不过于集中或分散
        """
        alpha_mean = alpha.mean(dim=-1, keepdim=True)  # 计算 alpha 的均值
        loss = torch.sum((alpha - alpha_mean) ** 2)  # 计算每个 alpha 与均值的差异
        return loss

    def importance_loss(self,alpha):
        """
        区域重要性奖励损失，提升 alpha 值大的区域的权重
        """
        # print(alpha)
        # print(torch.log(alpha + 1e-8))
        return -torch.sum(alpha * torch.log(alpha + 1e-8))  # 加入小的偏移量避免 log(0)

    def max_alpha_loss(self, alpha, beta=0.1):
        """
        最大化 alpha 最大值, 并给一个下限
        """
        # print(alpha.shape)
        alphas_part_max = torch.max(alpha, dim=0)[0]
        # print(alphas_part_max)
        return max(torch.Tensor([0]).cuda(), beta - alphas_part_max)



if __name__=="__main__":
    for i in range(20):
        
        x = torch.rand(32,7)
        loss = region_loss()
        y = loss(x)
        # print(y)