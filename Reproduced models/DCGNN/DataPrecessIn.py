from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io as scio
import numpy as np
import scipy.io as scio
import os
from metrics import Pearsoner_M
import scipy
import networkx as nx
import pandas as pd
from metrics import cal_grangercausality

def adjacency():
    row_ = np.array(
        [0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
         13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25, 25, 26, 26,
         27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40,
         41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54,
         54, 55, 55, 56, 57, 58, 59,
         60, 1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12,
         20, 13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26,
         34, 27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47,
         40, 48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58,
         54, 59, 55, 60, 56, 61, 61, 58, 59, 60, 61])

    col_ = np.array(
        [1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20,
         13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26, 34,
         27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47, 40,
         48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58, 54,
         59, 55, 60, 56, 61, 61, 58,
         59, 60, 61, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
         11, 12, 12, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25,
         25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38,
         39, 39, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52,
         53, 53, 54, 54, 55, 55, 56, 57, 58, 59, 60])

    # te_r = np.array([1,2,3])
    # tr_c = np.array([4,5,6])
    # data_ = np.ones(3).astype('float32')
    # B = scipy.sparse.csr_matrix((data_, (row_, col_)), shape=(62, 62))

    # weight_ = np.ones(236).astype('float32')
    # A = scipy.sparse.csr_matrix((weight_, (row_, col_)), shape=(62, 62))
    edges = pd.DataFrame()
    edges['sources'] = row_
    edges['targets'] = col_
    G = nx.from_pandas_edgelist(edges, source='sources',target='targets')
    A = nx.adjacency_matrix(G)
    A = A.todense()
    return A, G


    # return row_, col_, weight_ , A




class GraphGenerate1:
    def __init__(self,data_pre_graph, num_):
        self.num_ = num_
        self.data_pre_graph = data_pre_graph
        self.load_graph()
    def load_graph(self):
        feature_graph = self.data_pre_graph.transpose(1,0,2)
        # print(feature_graph.shape, self.num_, "GraphGenerate1")
        num_channel, num_size, num_feature = feature_graph.shape
        self.feature_graph = feature_graph.reshape(num_channel,num_size* num_feature)

        self.adj = Pearsoner_M(self.feature_graph, self.num_)


class GraphGenerate2:
    def __init__(self,data_pre_graph):
        self.data_pre_graph = data_pre_graph
        self.load_graph()
    def load_graph(self):
        feature_graph = self.data_pre_graph.transpose(1,0,2)
        num_channel, num_size, num_feature = feature_graph.shape
        self.feature_graph = feature_graph.reshape(num_channel,num_size* num_feature)

        self.adj = cal_grangercausality(self.feature_graph,self.feature_graph)


class GraphGenerate:
    def __init__(self,graphDir):
        self.graphDir = graphDir
        self.load_graph()

    def load_graph(self):
        graph_row   = scio.loadmat(self.graphDir)

        i = 0
        for key, value in graph_row.items():
            i += 1
            if i == 4:  # 去除前三项
                frature = np.asarray(value, dtype=np.float32)
                self.frature_graph = frature
            elif i > 4:
                frature = np.asarray(value, dtype=np.float32)
                self.frature_graph = np.concatenate([self.frature_graph, frature], axis=1)
        self.adj = Pearsoner_M(self.frature_graph,0.7)

        # for i in range(15):
        #     graph_name = 'djc_eeg' + str(i + 1)
        #     frature = np.asarray(graph_row[graph_name], dtype=np.float32)
        #     if i == 0:
        #         self.frature_graph = frature
        #     else:
        #         self.frature_graph   = np.concatenate([self.frature_graph,frature], axis = 1)
        # self.adj1  = Pearsoner_M(self.frature_graph)




class DataGenerate:
    def __init__(self, dataDir, labelDir, batch_size, timeStep, windowGap, pointer=0):
        self.dataDir      = dataDir
        self.labelDir     = labelDir
        self.batch_size   = batch_size
        self.timeStep     = timeStep
        self.windowGap    = windowGap
        self.pointer      = pointer
        self.load_data()
        # self.dataPreprocess()
        self.dataTimeProcess()
        self.batch = int(self.label_train.shape[0] / self.batch_size)
        # self.shuffle_data()

    def load_data(self):
        #按数据处理

        with open('%s'% self.labelDir,'r') as f_label:#导入label数据
            f_label_name = f_label.readline().strip()
            label_name = scio.loadmat(f_label_name)
            self.label = np.asarray(label_name['label'], dtype=np.float32)
            self.label[self.label == -1] = 2


            self.data     = [] # 所有数据存储到数据字典
            self.data_train   = [] # 个人训练数据初始化
            self.data_test    = [] # 个人测试数据初始化

            self.label_train  = []
            self.label_test   = []

            self.flag_train   = []
            self.flag_test    = []

            self.data_pre_graph = []

            data_row     = scio.loadmat(self.dataDir)
            # print(data.shape)
            for i in range(15): # 非夸人，前9个训练，后6个测试该操作设置flag，并整合所有数据，与label
                feature_name = 'de_LDS'+str(i+1)
                self.data         = np.asarray(data_row[feature_name], dtype=np.float32)



                self.data         = self.data.transpose(1,0,2) # 前两行转置
                if i == 0: # 堆积一个mat文件中的de数据,并且将生成每段数量相同的label矩阵
                    self.data_train   = self.data
                    # self.label_train  = self.label[0,i] * np.ones((len(self.data),1),dtype=np.int64)
                    self.flag_train   = i * np.ones((len(self.data),1),dtype=np.int64)
                elif i < 9:
                    self.data_train   = np.concatenate([self.data_train,self.data], axis = 0)

                    self.label_t      = self.label[0, i] * np.ones((len(self.data),1), dtype=np.int64)
                    # self.label_train  = np.concatenate([self.label_train,self.label_t])

                    self.flag_t       = i * np.ones((len(self.data),1), dtype=np.int64)
                    self.flag_train   = np.concatenate([self.flag_train,self.flag_t])

                elif i == 9:
                    self.data_test    = self.data
                    self.label_test   = self.label[0, i] * np.ones((len(self.data),1), dtype=np.int64)

                    self.flag_test    = i * np.ones((len(self.data),1), dtype=np.int64)

                else:
                    self.data_test    = np.concatenate([self.data_test, self.data], axis = 0)

                    self.label_t      = self.label[0, i] * np.ones((len(self.data),1), dtype=np.int64)
                    self.label_test   = np.concatenate([self.label_test,self.label_t])

                    self.flag_t       = i * np.ones((len(self.data),1), dtype=np.int64)
                    self.flag_test    = np.concatenate([self.flag_test, self.flag_t])


            self.data_pre_graph1  = self.data_train


            self.data_pre_graph = np.concatenate([self.data_train,self.data_test],axis=0)
            # #清空
            # self.data_train  = []
            # self.data_test   = []
            # self.label_train = []
            # self.label_test  = []
            # self.flag_train  = []
            # self.flag_test   = []


    def dataPreprocess(self):
        # self.complenet_train_data()
        nsam, ncha, nfea = self.data_train.shape
        mean_data        = np.mean(self.data_train.reshape(nsam * ncha, nfea), axis=0)
        std_data = np.std(self.data_train.reshape(nsam * ncha, nfea), axis=0)
        self.data_train = (self.data_train - mean_data) / std_data
        self.data_test = (self.data_test - mean_data) / std_data

    def dataTimeProcess(self):
        #按人构建时间的label与train/test set

        for i in range(15):
            start_train  = self.flag_train.squeeze().tolist()# 方便按照flag计数
            start_test   = self.flag_test.squeeze().tolist()
            if i < 9:
                start_num  = start_train.index(i) # 由于每个subject i都是独立的，所以不需要去考虑subject之间的连接
                start_flag = start_train.index(0)
                for n_time in range(start_num, start_num + np.sum(self.flag_train == i) - self.windowGap + self.timeStep):
                    if n_time == start_flag:
                        # python与matlab的矩阵计数有差距，要在后面补1
                        data_train_temp = self.data_train[np.newaxis, start_flag: self.windowGap - self.timeStep + 1, :, :]
                        continue
                    temp_a = self.data_train[np.newaxis, n_time:n_time + self.windowGap - self.timeStep + 1, :, :]
                    data_train_temp = np.concatenate((data_train_temp, temp_a), axis=0)

                label_train_temp = self.label[0,i] *np.ones((np.sum(self.flag_train == i)- self.windowGap + self.timeStep, 1),dtype=np.int64)


                if i == 0:
                    self.label_train = label_train_temp
                else:
                    self.label_train = np.concatenate([self.label_train,label_train_temp])

            else:
                start_num  = start_test.index(i)
                start_flag = start_test.index(9)
                for n_time in range(start_num, start_num + np.sum(self.flag_test == i) - self.windowGap + self.timeStep):
                    if n_time == start_flag:
                        # python与matlab的矩阵计数有差距，要在后面补1
                        data_test_temp = self.data_test[np.newaxis, start_flag: self.windowGap - self.timeStep + 1, :, :]
                        continue
                    temp_b = self.data_test[np.newaxis, n_time:n_time + self.windowGap - self.timeStep + 1, :, :]
                    data_test_temp = np.concatenate((data_test_temp, temp_b), axis=0)
                label_test_temp = self.label[0, i] * np.ones((np.sum(self.flag_test == i) - self.windowGap + self.timeStep, 1), dtype=np.int64)

                if i == 9:
                    self.label_test = label_test_temp
                else:
                    self.label_test = np.concatenate([self.label_test, label_test_temp])


        self.data_train = data_train_temp
        self.data_test  = data_test_temp


    def reset_pointer(self):
        self.pointer = 0
        # self.shuffle_data()

    def next_batch(self):
        data_batch = self.data_train[self.pointer : self.pointer+self.batch_size, :, :]
        label_batch = self.label_train[self.pointer : self.pointer+self.batch_size]
        self.pointer += self.batch_size
        if self.pointer >= self.data_train.shape[0]:
            self.reset_pointer()

        return data_batch, label_batch


