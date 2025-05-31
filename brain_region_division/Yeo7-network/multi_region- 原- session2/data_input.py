import numpy as np
import torch
import os
import scipy
from torch.utils.data import Dataset, DataLoader
import random
from scipy.ndimage import zoom
import math

# # 为 DataLoader 的每个工作线程设置随机种子
# def worker_init_fn(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# # 使用 torch.Generator 确保数据加载顺序一致
# g = torch.Generator()
# g.manual_seed(12)
# torch.random.manual_seed(g.initial_seed())

SEEDIV_data_dir = "../../../datasets/SEED-IV/eeg_feature_smooth/"

seesion1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
seesion2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
seesion3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels = [seesion1_label, seesion2_label, seesion3_label]

# 得到每个表情标签的数据索引
trials_seesion1 = [[3,5,6,8,20,22],[0,7,9,11,12,13],[1,4,10,14,16,17],[2,15,18,19,21,23]]#0,1,2,3±êÇ©Î»ÖÃ
trials_seesion2 = [[3,4,6,13,18,20],[1,14,15,17,21,23],[0,5,7,10,12,16],[2,8,9,11,19,22]]
trials_seesion3 = [[11,15,18,19,21,23],[0,3,7,8,10,22],[1,2,9,12,16,20],[4,5,6,13,14,17]]
trials = [trials_seesion1, trials_seesion2, trials_seesion3]
eeg_dict = {"train":[], "test":[]}


class  SEEDIV_dataset(Dataset):
    def __init__(self, SEEDIV_data_dir, subj_num, session_num, is_train):
        super(SEEDIV_dataset, self).__init__()
        self.is_train = is_train
        self.session_num = session_num
        # 候选测试样本
        self.test_Trails = []
        self.sample_list = []
        self.sample_label_list = []
        self.channel_group1 = [51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67] #11 5[63, 64, 65, 66, 67] 16
        self.channel_group2 = [18, 20, 26, 27, 28, 29, 30, 35, 36, 38, 39, 68, 69, 70, 71, 72]  #11 5[68, 69, 70, 71, 72] 16
        self.channel_group3 = [6, 7, 13, 14, 16, 17, 21, 22, 25, 31, 34, 40, 42, 43, 49, 50] #16
        self.channel_group4 = [15, 23, 24, 32, 33, 41, 73, 74, 75] #6 3 [73, 74, 75] 9
        self.channel_group5 = [1, 3, 4, 5] #4
        self.channel_group6 = [8, 9, 11, 12, 44, 45, 47, 48, 76] #8 1[76] 9 
        self.channel_group7 = [2, 10, 19, 37, 46, 54, 77, 78, 79] #6 3[77, 78, 79] 9
        self.channel_group_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
        #获取指定受试者的路径
        session_dir = os.path.join(SEEDIV_data_dir,str(self.session_num))
        for subj_file in os.listdir(session_dir):
            subj=int(subj_file.split("_")[0])
            if subj == subj_num:
                subj_file_name = subj_file
                break
        file_path = os.path.join(session_dir, subj_file_name)

        for i in range(4):
            self.test_Trails.append(trials[self.session_num-1][i][4])
            self.test_Trails.append(trials[self.session_num-1][i][5])
        # 读取de_LDS属性的数据，以此读取，避免标签对应不上
        for trial in range(24):
            X_de = scipy.io.loadmat(file_path)['de_LDS{}'.format(trial + 1)]
            # 首先进入对应session,然后再session里面获得对应标签
            y = int(labels[self.session_num-1][trial])
            # 从第二个维度进行划分，42个
            for t in range(X_de.shape[1]):
                x_de = torch.tensor(X_de[:, t, :]).float()
                # 单个样本数据预处理
                x_de = self.eeg_transformer(x_de,"Standardization")

                # 训练样本构建
                if self.is_train:
                   if not trial in self.test_Trails:
                        sample = {
                        'region1': self.getimage(x_de, self.channel_group1),
                            'region2':self.getimage(x_de, self.channel_group2),
                            'region3': self.getimage(x_de, self.channel_group3),
                            'region4': self.getimage(x_de, self.channel_group4),
                            'region5': self.getimage(x_de, self.channel_group5),
                            'region6': self.getimage(x_de, self.channel_group6),
                            'region7': self.getimage(x_de, self.channel_group7),
                            
                            'tran_fea1': self.gettran(x_de, self.channel_group1),
                            'tran_fea2': self.gettran(x_de, self.channel_group2),
                            'tran_fea3': self.gettran(x_de, self.channel_group3),
                            'tran_fea4': self.gettran(x_de, self.channel_group4),
                            'tran_fea5': self.gettran(x_de, self.channel_group5),
                            'tran_fea6': self.gettran(x_de, self.channel_group6),
                            'tran_fea7': self.gettran(x_de, self.channel_group7),

                            # 'region_all': self.getimage_all(x_de, self.channel_group_all),
                            # 'tran_fea_all':self.gettran(x_de, self.channel_group_all),
                        }
                        self.sample_list.append(sample)
                        self.sample_label_list.append(y)
                # 测试集构建
                else:
                   if trial in self.test_Trails:
                        sample = {
                        'region1': self.getimage(x_de, self.channel_group1),
                            'region2': self.getimage(x_de, self.channel_group2),
                            'region3': self.getimage(x_de, self.channel_group3),
                            'region4': self.getimage(x_de, self.channel_group4),
                            'region5': self.getimage(x_de, self.channel_group5),
                            'region6': self.getimage(x_de, self.channel_group6),
                            'region7': self.getimage(x_de, self.channel_group7),
                            
                            'tran_fea1': self.gettran(x_de, self.channel_group1),
                            'tran_fea2': self.gettran(x_de, self.channel_group2),
                            'tran_fea3': self.gettran(x_de, self.channel_group3),
                            'tran_fea4': self.gettran(x_de, self.channel_group4),
                            'tran_fea5': self.gettran(x_de, self.channel_group5),
                            'tran_fea6': self.gettran(x_de, self.channel_group6),
                            'tran_fea7': self.gettran(x_de, self.channel_group7),

                            # 'region_all': self.getimage_all(x_de, self.channel_group_all),
                            # 'tran_fea_all':self.gettran(x_de, self.channel_group_all),
                        }
                        self.sample_list.append(sample)
                        self.sample_label_list.append(y)
    
    def eeg_transformer(self, x, data_process):
        if data_process == "Standardization" :
            return (x - x.mean()) / x.std()
        # if data_process == "min_max_normalization":
        #     return (x - x.min()) / (x.max - x.min() + 1e-8)
        
    def __len__(self, ):
        return len(self.sample_list)

    def __getitem__(self, index):
        return self.sample_list[index], self.sample_label_list[index]


    def gettran(self, x, channel_selected):
        # print('channel_selected', channel_selected)
        d = torch.zeros(17,5)
        x = torch.cat((x,d), dim=0) # 多补一个0

        channel_selected = [x - 1 for x in channel_selected] 

        # print(len(channel_selected))
        return x[channel_selected, :]

    def getimage(self, x, channel_list):
        channel_selected = channel_list.copy()
        d = torch.zeros(17,5)
        x = torch.cat((x,d), dim=0) # 多补一个0

        # 计算需要的网格尺寸（开平方向上取整）
        n = len(channel_selected)
        grid_size = math.ceil(math.sqrt(n))
        image = torch.zeros(5, grid_size, grid_size)
        
        # 按行优先方式填充
        for idx, ch in enumerate(channel_selected):
            r, c = divmod(idx, grid_size)
            image[:, r, c] = x[ch - 1]

        scale_factors = (1, (grid_size*16)/grid_size, (grid_size*16)/grid_size)  # 每个通道的缩放因子，第一个维度保持不变
        # 对数据进行插值以调整维度
        image = zoom(image, scale_factors, order=1)  # order=1 代表线性插值

        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
        return image[:,:,:]


def getloader(SEEDIV_data_dir, subj_num=1, session_num=1, batch_size=8, is_train=False, is_drop_last=False, is_shuffle=True):
    gen_dataset = SEEDIV_dataset(SEEDIV_data_dir, subj_num, session_num, is_train)
    # print(True if is_train else False)
    # print(gen_dataset.__len__())
    return DataLoader(
            gen_dataset,
            batch_size=batch_size,
            shuffle = is_shuffle, 
            num_workers=0, 
            drop_last=is_drop_last,
            # worker_init_fn=worker_init_fn, 
            # generator=g
        )
    
if __name__== "__main__":
    import cv2
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_loader = getloader(SEEDIV_data_dir, subj_num=1, session_num=1, batch_size=32, is_train=True, is_drop_last=False, is_shuffle=True)
    
    for step, data in enumerate(data_loader, start=0):
        data, labels = data
        print(step, data['region1'].size(), data['region2'].size(),data['region3'].size(),data['region4'].size(),data['region5'].size(),
            data['region6'].size(),data['region7'].size(),data['tran_fea1'].size(),labels.size())
        break
