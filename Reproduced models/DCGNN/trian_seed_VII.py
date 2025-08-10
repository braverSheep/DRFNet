

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


# 整个训练脚本产生的结果都将存入这个脚本中，模型，log等等
def cre_prolog(num=1):
    # 获取当前脚本的绝对路径  
    current_script_path = os.path.abspath(__file__)  
    # 获取当前脚本的所在目录  
    parent_dir = os.path.dirname(current_script_path)  
    # 定义要创建的Pro_log目录的路径
    pro_log_dir_path = os.path.join(parent_dir, f'Pro_log_{num}')  
      
    # 检查Pro_log目录是否存在，如果不存在则创建  
    if not os.path.exists(pro_log_dir_path):  
        os.makedirs(pro_log_dir_path)  
        print(f"Directory '{pro_log_dir_path}' created.")  

    return pro_log_dir_path


def main(args):
    set_seeds(args.seed)

    fold_num = 1
    exp_dir = cre_prolog(fold_num)

    acc_all = []
    c_matrix_all=[]
    for t in range(20):  # Xtrain.shape[1]

        xtrain, ytrain, xtest, ytest = load_data_SEEDVII_dependent(subj_num=t+1, fold_num=fold_num)
        xtrain = torch.stack(xtrain).numpy().astype(np.float64)
        ytrain = np.array(ytrain).astype(np.int16)
        xtest = torch.stack(xtest).numpy().astype(np.float64)
        ytest = np.array(ytest).astype(np.int16)

        acc_t,c_matrix,all_max_train,all_max_test=trainer(args,t+1,xtrain,xtest,ytrain,ytest,exp_dir)
        acc_all.append(acc_t)
        c_matrix_all.append(c_matrix)

        print(acc_t,c_matrix)


    accFinal = acc_all
    i=0
    for ii in range(20):
        i+=1
        if i == 1:
            c_matrix_final = c_matrix_all[ii]
        else:
            c_matrix_final = c_matrix_final+ c_matrix_all[ii]
    c_matrix_final=c_matrix_final/c_matrix_final.sum(1)[:,None]
    acc_mean = np.mean(accFinal)
    acc_std = np.std(accFinal)
    print(acc_all,"\n")
    print(c_matrix_final,"\n")
    print(acc_mean, acc_std)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  type=int,
                        default=12,
                        help='seed')
    parser.add_argument('--batch_size',  type=int,
                        default=32,
                        help='batch_size')
    parser.add_argument('--total_epoch',  type=int,
                        default=180,
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
                        default=7,
                        help='num of class')
    parser.add_argument('--prebn', type=bool,
                        default=False,
                        help='prebn or not')
    parser.add_argument('--save', type=bool,
                        default=True,
                        help='num of class')
    args = parser.parse_args()
    main(args)
