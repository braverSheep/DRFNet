import numpy as np
import torch
import os
import scipy
from torch.utils.data import Dataset, DataLoader
import random
from scipy.ndimage import zoom

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
    def __init__(self, SEEDIV_data_dir, subj_num, session_num, is_train, cross_validation):
        super(SEEDIV_dataset, self).__init__()
        self.is_train = is_train
        self.session_num = session_num
        # 候选测试样本
        self.test_Trails = []
        self.sample_list = []
        self.sample_label_list = []
        self.channel_group1 = [2 ,1 ,3, 4, 63, 5, 60, 61, 62] # 63是自己杜撰的空白点
        self.channel_group2 = [6 ,7 ,8, 15, 16, 17, 24, 25, 26 ]
        self.channel_group3 = [24, 25, 26, 33, 34, 35, 42, 43, 44]
        self.channel_group4 = [12, 13, 14, 21, 22, 23, 30, 31, 32]
        self.channel_group5 = [30, 31, 32, 39, 40, 41, 48, 49, 50]
        self.channel_group6 = [ 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 
                                26, 27, 28, 29, 30, 35, 36, 37, 38, 39,
                                44, 45, 46, 47, 48]
        self.channel_group7 = [51, 52, 53, 54, 55, 56, 57, 58, 59] #, 60, 61, 62
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
            self.test_Trails.append(trials[self.session_num-1][i][cross_validation]) # cross_validation=0/2/4
            self.test_Trails.append(trials[self.session_num-1][i][cross_validation+1])
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
                        sample = {'region1': self.getimage(x_de, self.channel_group1),
                            'region2': self.getimage(x_de, self.channel_group2),
                            'region3': self.getimage(x_de, self.channel_group3),
                            'region4': self.getimage(x_de, self.channel_group4),
                            'region5': self.getimage(x_de, self.channel_group5),
                            'region6': self.getimage_big(x_de, self.channel_group6),
                            'region7': self.getimage(x_de, self.channel_group7),
                            
                            'tran_fea1': self.gettran(x_de, self.channel_group1),
                            'tran_fea2': self.gettran(x_de, self.channel_group2),
                            'tran_fea3': self.gettran(x_de, self.channel_group3),
                            'tran_fea4': self.gettran(x_de, self.channel_group4),
                            'tran_fea5': self.gettran(x_de, self.channel_group5),
                            'tran_fea6': self.gettran(x_de, self.channel_group6),
                            'tran_fea7': self.gettran(x_de, self.channel_group7),

                            'region_all': self.getimage_all(x_de, self.channel_group_all),
                            'tran_fea_all':self.gettran(x_de, self.channel_group_all),
                        }
                        self.sample_list.append(sample)
                        self.sample_label_list.append(y)
                # 测试集构建
                else:
                   if trial in self.test_Trails:
                        sample = {'region1': self.getimage(x_de, self.channel_group1),
                            'region2': self.getimage(x_de, self.channel_group2),
                            'region3': self.getimage(x_de, self.channel_group3),
                            'region4': self.getimage(x_de, self.channel_group4),
                            'region5': self.getimage(x_de, self.channel_group5),
                            'region6': self.getimage_big(x_de, self.channel_group6),
                            'region7': self.getimage(x_de, self.channel_group7),
                            
                            'tran_fea1': self.gettran(x_de, self.channel_group1),
                            'tran_fea2': self.gettran(x_de, self.channel_group2),
                            'tran_fea3': self.gettran(x_de, self.channel_group3),
                            'tran_fea4': self.gettran(x_de, self.channel_group4),
                            'tran_fea5': self.gettran(x_de, self.channel_group5),
                            'tran_fea6': self.gettran(x_de, self.channel_group6),
                            'tran_fea7': self.gettran(x_de, self.channel_group7),

                            'region_all': self.getimage_all(x_de, self.channel_group_all),
                            'tran_fea_all':self.gettran(x_de, self.channel_group_all),
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
        d = torch.zeros(1,5)
        x = torch.cat((x,d), dim=0) # 多补一个0

        channel_selected = [x - 1 for x in channel_selected] 

        # print(len(channel_selected))
        return x[channel_selected, :]

    def getimage(self, x, channelist):
        channel_selected = channelist.copy()
        d = torch.zeros(1,5)
        x = torch.cat((x,d), dim=0) # 多补一个0

        image = torch.zeros(5, 3, 3)
        
        if len(channel_selected) < 9:
            for i in range(9-len(channel_selected)):
                channel_selected.append(62)

        for i in range(5):
            image[i, 0, 0] = x[channel_selected[0]-1, i]
            image[i, 0, 1] = x[channel_selected[1]-1, i]
            image[i, 0, 2] = x[channel_selected[2]-1, i]

            image[i, 1, 0] = x[channel_selected[3]-1, i]
            image[i, 1, 1] = x[channel_selected[4]-1, i]
            image[i, 1, 2] = x[channel_selected[5]-1, i]

            image[i, 2, 0] = x[channel_selected[6]-1, i]
            image[i, 2, 1] = x[channel_selected[7]-1, i]
            image[i, 2, 2] = x[channel_selected[8]-1, i]
            

        scale_factors = (1, 48/3, 48/3)  # 每个通道的缩放因子，第一个维度保持不变
        # 对数据进行插值以调整维度
        image = zoom(image, scale_factors, order=1)  # order=1 代表线性插值

        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
        return image[:,:,:]

    def getimage_big(self, x, channel_selected):
        
        image = torch.zeros(5, 5, 5)

        for i in range(5):
            image[i, 0, 0] = x[channel_selected[0]-1, i]
            image[i, 0, 1] = x[channel_selected[1]-1, i]
            image[i, 0, 2] = x[channel_selected[2]-1, i]
            image[i, 0, 3] = x[channel_selected[3]-1, i]
            image[i, 0, 4] = x[channel_selected[4]-1, i]

            image[i, 1, 0] = x[channel_selected[5]-1, i]
            image[i, 1, 1] = x[channel_selected[6]-1, i]
            image[i, 1, 2] = x[channel_selected[7]-1, i]
            image[i, 1, 3] = x[channel_selected[8]-1, i]
            image[i, 1, 4] = x[channel_selected[9]-1, i]

            image[i, 2, 0] = x[channel_selected[10]-1, i]
            image[i, 2, 1] = x[channel_selected[11]-1, i]
            image[i, 2, 2] = x[channel_selected[12]-1, i]
            image[i, 2, 3] = x[channel_selected[13]-1, i]
            image[i, 2, 4] = x[channel_selected[14]-1, i]

            image[i, 3, 0] = x[channel_selected[15]-1, i]
            image[i, 3, 1] = x[channel_selected[16]-1, i]
            image[i, 3, 2] = x[channel_selected[17]-1, i]
            image[i, 3, 3] = x[channel_selected[18]-1, i]
            image[i, 3, 4] = x[channel_selected[19]-1, i]

            image[i, 4, 0] = x[channel_selected[20]-1, i]
            image[i, 4, 1] = x[channel_selected[21]-1, i]
            image[i, 4, 2] = x[channel_selected[22]-1, i]
            image[i, 4, 3] = x[channel_selected[23]-1, i]
            image[i, 4, 4] = x[channel_selected[24]-1, i]
            
        scale_factors = (1, 80 / 5, 80 / 5)  # 每个通道的缩放因子，第一个维度保持不变
        # 对数据进行插值以调整维度
        image = zoom(image, scale_factors, order=1)  # order=1 代表线性插值

        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
        return image[:,:,:]

    def getimage_all(self, x, nn):
        channel_selected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
        image = torch.zeros(5, 9, 9)

        for i in range(5):
            image[i, 0, 3] = x[channel_selected[0], i]
            image[i, 0, 4] = x[channel_selected[1], i]
            image[i, 0, 5] = x[channel_selected[2], i]

            image[i, 1, 3] = x[channel_selected[3], i]
            image[i, 1, 5] = x[channel_selected[4], i]

            image[i, 2, 0] = x[channel_selected[5], i]
            image[i, 2, 1] = x[channel_selected[6], i]
            image[i, 2, 2] = x[channel_selected[7], i]
            image[i, 2, 3] = x[channel_selected[8], i]
            image[i, 2, 4] = x[channel_selected[9], i]
            image[i, 2, 5] = x[channel_selected[10], i]
            image[i, 2, 6] = x[channel_selected[11], i]
            image[i, 2, 7] = x[channel_selected[12], i]
            image[i, 2, 8] = x[channel_selected[13], i]

            image[i, 3, 0] = x[channel_selected[14], i]
            image[i, 3, 1] = x[channel_selected[15], i]
            image[i, 3, 2] = x[channel_selected[16], i]
            image[i, 3, 3] = x[channel_selected[17], i]
            image[i, 3, 4] = x[channel_selected[18], i]
            image[i, 3, 5] = x[channel_selected[19], i]
            image[i, 3, 6] = x[channel_selected[20], i]
            image[i, 3, 7] = x[channel_selected[21], i]
            image[i, 3, 8] = x[channel_selected[22], i]

            image[i, 4, 0] = x[channel_selected[23], i]
            image[i, 4, 1] = x[channel_selected[24], i]
            image[i, 4, 2] = x[channel_selected[25], i]
            image[i, 4, 3] = x[channel_selected[26], i]
            image[i, 4, 4] = x[channel_selected[27], i]
            image[i, 4, 5] = x[channel_selected[28], i]
            image[i, 4, 6] = x[channel_selected[29], i]
            image[i, 4, 7] = x[channel_selected[30], i]
            image[i, 4, 8] = x[channel_selected[31], i]

            image[i, 5, 0] = x[channel_selected[32], i]
            image[i, 5, 1] = x[channel_selected[33], i]
            image[i, 5, 2] = x[channel_selected[34], i]
            image[i, 5, 3] = x[channel_selected[35], i]
            image[i, 5, 4] = x[channel_selected[36], i]
            image[i, 5, 5] = x[channel_selected[37], i]
            image[i, 5, 6] = x[channel_selected[38], i]
            image[i, 5, 7] = x[channel_selected[39], i]
            image[i, 5, 8] = x[channel_selected[40], i]

            image[i, 6, 0] = x[channel_selected[41], i]
            image[i, 6, 1] = x[channel_selected[42], i]
            image[i, 6, 2] = x[channel_selected[43], i]
            image[i, 6, 3] = x[channel_selected[44], i]
            image[i, 6, 4] = x[channel_selected[45], i]
            image[i, 6, 5] = x[channel_selected[46], i]
            image[i, 6, 6] = x[channel_selected[47], i]
            image[i, 6, 7] = x[channel_selected[48], i]
            image[i, 6, 8] = x[channel_selected[49], i]

            image[i, 7, 1] = x[channel_selected[50], i]
            image[i, 7, 2] = x[channel_selected[51], i]
            image[i, 7, 3] = x[channel_selected[52], i]
            image[i, 7, 4] = x[channel_selected[53], i]
            image[i, 7, 5] = x[channel_selected[54], i]
            image[i, 7, 6] = x[channel_selected[55], i]
            image[i, 7, 7] = x[channel_selected[56], i]

            image[i, 8, 2] = x[channel_selected[57], i]
            image[i, 8, 3] = x[channel_selected[58], i]
            image[i, 8, 4] = x[channel_selected[59], i]
            image[i, 8, 5] = x[channel_selected[60], i]
            image[i, 8, 6] = x[channel_selected[61], i]

        scale_factors = (1, 48 / 9, 48 / 9)  # 每个通道的缩放因子，第一个维度保持不变
        # 对数据进行插值以调整维度
        image = zoom(image, scale_factors, order=1)  # order=1 代表线性插值
        # image 形状改为 (5, 112, 112)

        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
        return image[:,:,:]


def getloader(SEEDIV_data_dir, subj_num=1, session_num=1, batch_size=8, is_train=False, is_drop_last=False, is_shuffle=True, cross_validation=0):
    gen_dataset = SEEDIV_dataset(SEEDIV_data_dir, subj_num, session_num, is_train, cross_validation)
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
    data_loader = getloader(SEEDIV_data_dir, subj_num=1, session_num=1, batch_size=32, is_train=True, is_drop_last=False, is_shuffle=True, cross_validation=0)
    
    for step, data in enumerate(data_loader, start=0):
        data, labels = data
        print(step, data['region1'].size(), data['region2'].size(),data['region3'].size(),data['region4'].size(),data['region5'].size(),
            data['region6'].size(),data['region7'].size(),data['tran_fea1'].size(),labels.size())
        break
