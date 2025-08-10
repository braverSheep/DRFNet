from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import random

import torch


from data import *
from train import trainer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_seeds(args.seed)
    feature_data1,feature_data2, label1,label2 = load_data_FACED_independent(flag=3)

    acc_all = []
    c_matrix_all=[]
    for t in range(10):  # Xtrain.shape[1]
        if t <9:
            xtrain = np.array([feature_data1[i] for i in range(9) if i not in [t] ])
            xtrain = xtrain.reshape(xtrain.shape[0] * xtrain.shape[1], xtrain.shape[2], xtrain.shape[3])
            xtrain = np.append(xtrain,feature_data2,axis=0).astype(np.float64)
            xtest  = np.array(feature_data1[t]).astype(np.float64)
            ytrain = np.tile(label1,8)
            ytrain = np.append(ytrain,label2,axis=0).astype(np.int16)
            ytest  = label1.astype(np.int16)

        else:
            xtrain = np.array([feature_data1[i] for i in range(9)])
            xtrain = xtrain.reshape(xtrain.shape[0] * xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]).astype(np.float64)

            xtest  = np.array(feature_data2).astype(np.float64)
            ytrain = np.tile(label1, 9).astype(np.int16)
            ytest  = label2.astype(np.int16)




        acc_t,c_matrix,all_max_train,all_max_test=trainer(args,t+1,xtrain,xtest,ytrain,ytest)
        acc_all.append(acc_t)
        c_matrix_all.append(c_matrix)
        if args.save:
            scio.savemat('FACED/S'+str(t+1)+'.mat', {'A_all_test': all_max_test[0].cpu().numpy(),
                                                    'L_all_test': all_max_test[1].cpu().numpy(),
                                                    'M_all_test': all_max_test[2].cpu().numpy(),
                                                    })
        print(acc_t,c_matrix)


    accFinal = acc_all
    i=0
    for ii in range(10):
        i+=1
        if i == 1:
            c_matrix_final = c_matrix_all[ii]
        else:
            c_matrix_final = c_matrix_final+ c_matrix_all[ii]
    c_matrix_final=c_matrix_final/c_matrix_final.sum(1)[:,None]
    acc_mean = np.mean(accFinal)
    acc_std = np.std(accFinal)
    print(acc_all)
    print(c_matrix_final)
    print(acc_mean, acc_std)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  type=int,
                        default=123,
                        help='seed')
    parser.add_argument('--batch_size',  type=int,
                        default=32,
                        help='batch_size')
    parser.add_argument('--total_epoch',  type=int,
                        default=100,
                        help='number of epochs of the training process')
    parser.add_argument('--learning_rate',     type=float,
                        default=0.001,
                        help='learning_rate')
    parser.add_argument('--dropout_rate', type=float,
                        default=0.5,
                        help='dropout_rate')
    parser.add_argument('--weight_decay_rate', type=float,
                        default=0.001,
                        help='the rate of weight decay')
    parser.add_argument('--K', type=int,
                        default=2,
                        help='number of layers of gcn')
    parser.add_argument('--lambdaa', type=int,
                        default=0.1,
                        help='the rate of part')
    parser.add_argument('--num_out', type=int,
                        default=32,
                        help='output size of gcn')
    parser.add_argument('--nclass', type=int,
                        default=3,
                        help='num of class')
    parser.add_argument('--prebn', type=bool,
                        default=False,
                        help='prebn or not')
    parser.add_argument('--save', type=bool,
                        default=True,
                        help='num of class')
    args = parser.parse_args()
    main(args)
