
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

SEEDIV_data_dir = r"D:/research/datasets/SEED-VII/EEG_features/"


# session1_trial = 
# [0, 1, 2, 3, 4,   5, 6, 7, 8, 9,    10, 11, 12, 13, 14,   15, 16, 17, 18, 19, ]
# [20, 21, 22, 23, 24,   25, 26, 27, 28, 29,   30, 31, 32, 33, 34,   35, 36, 37, 38, 39,]
# [40, 41, 42, 43, 44,   45, 46, 47, 48, 49,   50, 51, 52, 53, 54,   55, 56, 57, 58, 59,]
# [60, 61, 62, 63, 64,   65, 66, 67, 68, 69,   70, 71, 72, 73, 74,   75, 76, 77, 78, 79,]
labels = [0,   2,   3,   5,   6,   6,   5,   3,   2,   0,   0,   2,   3,   5,   6,   6,   5,   3,   2,   0,\
          6,   5,   4,   2,   1,   1,   2,   4,   5,   6,   6,   5,   4,   2,   1,   1,   2,   4,   5,   6,\
          0,   1,   3,   4,   6,   6,   4,   3,   1,   0,   0,   1,   3,   4,   6,   6,   4,   3,   1,   0,\
          3,   5,   4,   1,   0,   0,   1,   4,   5,   3,   3,   5,   4,   1,   0,   0,   1,   4,   5,   3,\
]

# -----------------4 fold cross validation-----------
train_trials_1 = [0, 1, 2, 3, 4,   5, 6, 7, 8, 9,    10, 11, 12, 13, 14,\
                20, 21, 22, 23, 24,   25, 26, 27, 28, 29,   30, 31, 32, 33, 34,\
                40, 41, 42, 43, 44,   45, 46, 47, 48, 49,   50, 51, 52, 53, 54,\
                60, 61, 62, 63, 64,   65, 66, 67, 68, 69,   70, 71, 72, 73, 74,
                ]
test_trails_1 = [15, 16, 17, 18, 19,\
                35, 36, 37, 38, 39,\
                55, 56, 57, 58, 59,\
                75, 76, 77, 78, 79,
                ]

# -----------------4 fold cross validation-----------
train_trials_2 = [0, 1, 2, 3, 4,   5, 6, 7, 8, 9,    15, 16, 17, 18, 19,\
                20, 21, 22, 23, 24,   25, 26, 27, 28, 29,   35, 36, 37, 38, 39,\
                40, 41, 42, 43, 44,   45, 46, 47, 48, 49,   55, 56, 57, 58, 59,\
                60, 61, 62, 63, 64,   65, 66, 67, 68, 69,   75, 76, 77, 78, 79,
                ]
test_trails_2 = [10, 11, 12, 13, 14,\
                30, 31, 32, 33, 34,\
                50, 51, 52, 53, 54,\
                70, 71, 72, 73, 74,
                ]

# -----------------4 fold cross validation-----------
train_trials_3 = [0, 1, 2, 3, 4,   15, 16, 17, 18, 19,    10, 11, 12, 13, 14,\
                20, 21, 22, 23, 24,   35, 36, 37, 38, 39,   30, 31, 32, 33, 34,\
                40, 41, 42, 43, 44,   55, 56, 57, 58, 59,   50, 51, 52, 53, 54,\
                60, 61, 62, 63, 64,   75, 76, 77, 78, 79,   70, 71, 72, 73, 74,
                ]
test_trails_3 = [5, 6, 7, 8, 9,\
                25, 26, 27, 28, 29,\
                45, 46, 47, 48, 49,\
                65, 66, 67, 68, 69,
                ]

# -----------------4 fold cross validation-----------
train_trials_4 = [15, 16, 17, 18, 19,   5, 6, 7, 8, 9,    10, 11, 12, 13, 14,\
                35, 36, 37, 38, 39,   25, 26, 27, 28, 29,   30, 31, 32, 33, 34,\
                55, 56, 57, 58, 59,   45, 46, 47, 48, 49,   50, 51, 52, 53, 54,\
                75, 76, 77, 78, 79,   65, 66, 67, 68, 69,   70, 71, 72, 73, 74,
                ]
test_trails_4 = [0, 1, 2, 3, 4,\
                20, 21, 22, 23, 24,\
                40, 41, 42, 43, 44,\
                60, 61, 62, 63, 64,
                ]

three_fold_train = [train_trials_1, train_trials_2, train_trials_3, train_trials_4]
three_fold_test = [test_trails_1, test_trails_2, test_trails_3, test_trails_4]







def getsample_t(data, T=6):
    # data = np.zeros((6, 62, 5))  # 这里只是示例，换成你的真实 data
    # data[0][0][0] = 1
    # data[0][0][1] = 1
    # data[0][0][2] = 1
    # data[0][0][3] = 1
    # data[0][0][4] = 1


    # data[0][-1][0] = 1
    # data[0][-1][1] = 1
    # data[0][-1][2] = 1
    # data[0][-1][3] = 1
    # data[0][-1][4] = 1

    # data[0][27][0] = 1
    # data[0][27][1] = 1
    # data[0][27][2] = 1
    # data[0][27][3] = 1
    # data[0][27][4] = 1

    # print(data)

    # 电极名称列表（长度 62），与 data 的第 0 维一一对应
    names = [
        "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
        "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4","C6","T8",
        "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
        "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
        "PO7","PO5","PO3","POZ","PO4","PO6","PO8",
        "CB1","O1","OZ","O2","CB2"
    ]

    # 9x9 的电极位置布局（用 "0" 表示该位置没有电极）
    grid = [
        ["0","0","0","FP1","FPZ","FP2","0","0","0"],
        ["0","0","0","AF3","0","AF4","0","0","0"],
        ["F7","F5","F3","F1","FZ","F2","F4","F6","F8"],
        ["FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8"],
        ["T7","C5","C3","C1","CZ","C2","C4","C6","T8"],
        ["TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8"],
        ["P7","P5","P3","P1","PZ","P2","P4","P6","P8"],
        ["0","PO7","PO5","PO3","POZ","PO4","PO6","PO8","0"],
        ["0","0","CB1","O1","OZ","O2","CB2","0","0"],
    ]

    # 2) 名称到索引的字典
    name_to_idx = {name: i for i, name in enumerate(names)}

    # 3) 构造 idx_map_p：shape = (9,9)，无电极处为 -1，然后全部 +1 → (-1→0, 真正索引 i→i+1)
    idx_map = torch.tensor([
        [ name_to_idx[label] if label != "0" else -1 for label in row ]
        for row in grid
    ], dtype=torch.int64, device=data.device)
    idx_map_p = idx_map + 1  # 这样 0 则映射到 data_padded[ :, 0, : ]（全 0 行）

    # 4) 在第 1 维（电极维）前面补一行全 0：
    #    data_padded.shape = (T, 63, 5)
    T, _, C = data.shape
    pad = torch.zeros((T, 1, C), dtype=data.dtype, device=data.device)
    data_padded = torch.cat([pad, data], dim=1)

    # 5) 高级索引一次拿到 (T, 9, 9, 5)
    #    data_padded: (T, 63, 5)
    #    idx_map_p:    (9, 9)
    # PyTorch 会把 idx_map_p 广播到 batch 维度：
    grid_data = data_padded[:, idx_map_p, :]  # -> (T, 9, 9, 5)

    # 6) 维度换到 (T, 5, 9, 9)
    out = grid_data.permute(0, 3, 1, 2).contiguous()

    return out


class  SEEDV_dataset(Dataset):
    def __init__(self, SEEDV_data_dir, subj_num, fold_num=1, is_train=True, T=5):
        super(SEEDV_dataset, self).__init__()
        self.is_train = is_train
        self.fold_num = fold_num - 1
        # 候选测试样本
        self.test_Trails = []
        self.sample_list = []
        self.sample_label_list = []
        

        #获取指定受试者的路径
        for subj_file in os.listdir(SEEDV_data_dir):
            subj=int(subj_file.split(".")[0])
            if subj == subj_num:
                subj_file_name = subj_file
                break
        file_path = os.path.join(SEEDV_data_dir, subj_file_name)
        print(file_path)


        if is_train:
            print(three_fold_train[self.fold_num])
            for trial in three_fold_train[self.fold_num]:
                X_de = scipy.io.loadmat(file_path)['de_LDS_{}'.format(trial + 1)]
                for t in range(X_de.shape[0]-T):
                    x_de = torch.tensor(X_de[t:t+T, :, :]).float() # [5,62] 需要转换成【62,5】
                    x_de = x_de.permute(0,2,1)
                    # print(x_de.shape)
                    # 单个样本数据预处理
                    x = self.eeg_transformer(x_de,"Standardization")
                    y = int(labels[trial])

                    self.sample_list.append(getsample_t(x, T))
                    self.sample_label_list.append(y)
                    
                
        else:
            print(three_fold_test[self.fold_num])
            for trial in three_fold_test[self.fold_num]:
                X_de = scipy.io.loadmat(file_path)['de_LDS_{}'.format(trial + 1)]
                for t in range(X_de.shape[0]-T):
                    x_de = torch.tensor(X_de[t:t+T, :, :]).float() # [5,62] 需要转换成【62,5】
                    x_de = x_de.permute(0,2,1)
                    # print(x_de.shape)
                    # 单个样本数据预处理
                    x = self.eeg_transformer(x_de,"Standardization")
                    y = int(labels[trial])

                    self.sample_list.append(getsample_t(x, T))
                    self.sample_label_list.append(y)
        pass


    def eeg_transformer(self, x, data_process):#标准化处理
        if data_process == "Standardization" :
            return (x - x.mean()) / x.std()
        # if data_process == "min_max_normalization":
        #     return (x - x.min()) / (x.max - x.min() + 1e-8)
        
    def __len__(self, ):
        return len(self.sample_list)

    def __getitem__(self, index):#返回索引
        return self.sample_list[index], self.sample_label_list[index]


    def gettran(self, x, channel_selected):#从输入数据 x 中选择指定的通道
        # print('channel_selected', len(channel_selected))
        channel_selected = [x - 1 for x in channel_selected]
        # print(len(channel_selected))
        return x[channel_selected, :]



def getloader(SEEDV_data_dir, subj_num=1, fold_num=0, batch_size=8, is_train=False, is_drop_last=False, is_shuffle=True):
    gen_dataset = SEEDV_dataset(SEEDV_data_dir, subj_num, fold_num, is_train)
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
    data_loader = getloader(SEEDIV_data_dir, subj_num=1, fold_num=1, batch_size=32, is_train=False, is_drop_last=False, is_shuffle=True)
    
    for step, data in enumerate(data_loader, start=0):
        data, labels = data
        print(step, data.size())
        break
    print(len(data_loader.dataset))
    