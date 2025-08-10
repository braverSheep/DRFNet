# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import pickle as pkl
import pickle
import os
import torch
import scipy

def load_data_FACED_independent(flag=1,flag_band=5,channel_t=1):
    features = ['psd', 'de']
    feature_data = np.zeros(shape=(1,10),dtype=object)
    subs_de = np.zeros(shape=(123, 28, 32, 30, 5),dtype=object)
    for i in range(123):  # 123 because range is exclusive at the end
        # Format the filename with leading zeros
        filename = f'sub{i:03}.pkl.pkl'
        # Open and load the pickle file
        with open('FACED/DE/' + filename, 'rb') as f:
          subs_de[i] = pkl.load(f)
    groups = [subs_de[i*12:(i+1)*12] for i in range(9)]
    groups_last = subs_de[108:]
    reshaped_groups = [group.reshape(-1, 32, 5) for group in groups]
    reshaped_last_group = groups_last.reshape(-1, 32, 5)
    if flag == 3:
        labels = []
        for i in range(3):
            if i == 1:
                labels.extend([i] * 4)  
            else:
                labels.extend([i] * 12)  
        labels = np.array(labels)
        expanded_labels = np.repeat(labels, 30)  # (28 * 30,)
        final_labels1 = np.tile(expanded_labels, (12, 1))  # (12 * 28 * 30, 1)
        final_labels2 = np.tile(expanded_labels, (15, 1))  # (15 * 28 * 30, 1)


        final_labels1 = final_labels1.flatten()
        final_labels2 = final_labels2.flatten()
    if flag == 9:
        labels = []
        for i in range(9):
            if i == 1:
                labels.extend([i] * 4)  
            else:
                labels.extend([i] * 3)  


        labels = np.array(labels)

        expanded_labels = np.repeat(labels, 30)  # (28 * 30,)


        final_labels1 = np.tile(expanded_labels, (12, 1))  # (12 * 28 * 30, 1)
        final_labels2 = np.tile(expanded_labels, (15, 1))  # (15 * 28 * 30, 1)


        final_labels1 = final_labels1.flatten()
        final_labels2 = final_labels2.flatten()

    return reshaped_groups,reshaped_last_group,final_labels1,final_labels2

def load_data_SEEDV_independent(flag=1):
    subs_de = np.zeros((16,), dtype=object)
    feature_data = np.zeros(shape=(1, 16), dtype=object)
    for i in range(16):
        filename = f'D:/research/datasets/SEED-V/EEG_DE_features/{i + 1}_123.npz'  
        with np.load(filename, 'rb') as f:  
            subs_de[i] = {key: f[key] for key in f.files}  
            data = pickle.loads(subs_de[i]['data'])
            label = pickle.loads(subs_de[i]['label'])
            data_de = np.concatenate(list(data.values()), axis=0)
            print(data_de.shape)
            label_de = np.concatenate(list(label.values()), axis=0)
            data_de = data_de.reshape(-1,62,5)
            print(data_de.shape)
            feature_data[0,i] = data_de
    print(feature_data.shape)
    return feature_data, label_de



def eeg_transformer(x, data_process):
    if data_process == "Standardization" :
        return (x - x.mean()) / x.std()

def load_data_SEEDVII_dependent(subj_num=1, fold_num=1):
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

    xtrain, ytrain, xtest, ytest = [], [], [], []
    
    SEEDVII_data_dir = r"D:\research\datasets\SEED-VII\EEG_features"

    #获取指定受试者的路径
    for subj_file in os.listdir(SEEDVII_data_dir):
        subj=int(subj_file.split(".")[0])
        if subj == subj_num:
            subj_file_name = subj_file
            break
    file_path = os.path.join(SEEDVII_data_dir, subj_file_name)
    print(file_path)


    print(three_fold_train[fold_num-1])
    for trial in three_fold_train[fold_num-1]:
        X_de = scipy.io.loadmat(file_path)['de_LDS_{}'.format(trial + 1)]
        for t in range(X_de.shape[0]):
            x_de = torch.tensor(X_de[t, :, :]).float() # [5,62] 需要转换成【62,5】
            x_de = x_de.permute(1,0)

            # 单个样本数据预处理
            x_de = eeg_transformer(x_de,"Standardization")
            y = int(labels[trial])
            
            xtrain.append(x_de)
            ytrain.append(y)


    print(three_fold_test[fold_num-1])
    for trial in three_fold_test[fold_num-1]:
        X_de = scipy.io.loadmat(file_path)['de_LDS_{}'.format(trial + 1)]
        for t in range(X_de.shape[0]):
            x_de = torch.tensor(X_de[t, :, :]).float() # [5,62] 需要转换成【62,5】
            # print(x_de.shape)
            x_de = x_de.permute(1,0)
            # print(x_de.shape)

            # 单个样本数据预处理
            x_de = eeg_transformer(x_de,"Standardization")
            y = int(labels[trial])

            xtest.append(x_de)
            ytest.append(y)


    return xtrain, ytrain, xtest, ytest

if __name__ == "__main__":
    # xtrain, ytrain, xtest, ytest = load_data_SEEDV_dependent()
    # # 如果 xtrain 是 List[Tensor]，先转换为 numpy，再堆叠
    # xtrain = [x.numpy() if isinstance(x, torch.Tensor) else x for x in xtrain]

    # # 检查每个元素的 shape 是否一致
    # shapes = [x.shape for x in xtrain]
    # print(set(shapes))  # 如果只返回一个 shape 就说明一致
    # print(len(xtrain), len(xtest))

    xtrain, ytrain, xtest, ytest = load_data_SEEDVII_dependent()
    # 如果 xtrain 是 List[Tensor]，先转换为 numpy，再堆叠
    xtrain = [x.numpy() if isinstance(x, torch.Tensor) else x for x in xtrain]

    # 检查每个元素的 shape 是否一致
    shapes = [x.shape for x in xtrain]
    print(set(shapes))  # 如果只返回一个 shape 就说明一致
    print(len(xtrain), len(xtest))