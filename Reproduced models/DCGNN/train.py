import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
import time
from model import *
from utils import *
from cdt.causality.graph import GES, PC, GIES
from DataPrecessIn import DataGenerate, GraphGenerate, GraphGenerate1, adjacency, GraphGenerate2
import pandas as pd
import networkx as nx
import numpy as np
from metrics import cal_grangercausality
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import cycle
from torch.autograd import Variable
import os




def trainer(args, t, xtrain, xtest, ytrain, ytest,exp_dir):
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    obj = GES()

    print("xtrain.shape[0]", xtrain.shape[0])
    graph = GraphGenerate1(xtrain, int(xtrain.shape[1] * xtrain.shape[1] * 0.10))
    data1 = xtrain.transpose(0, 2, 1).reshape(xtrain.shape[0] * xtrain.shape[2], 62)
    data1 = pd.DataFrame(data1)
    output = obj.predict(data1, nx.Graph(graph.adj))
    adj_pa = np.asarray(nx.adjacency_matrix(output).todense()).astype('float32')
    adj_pa = torch.tensor(adj_pa).to(0)

    xtrain = torch.from_numpy(xtrain)
    xtest = torch.from_numpy(xtest)
    ytrain = torch.from_numpy(ytrain)
    ytest = torch.from_numpy(ytest)

    train_dataset = Data.TensorDataset(xtrain, ytrain)
    test_dataset = Data.TensorDataset(xtest, ytest)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    model = GNN(xtrain.shape, args.K, args.num_out, args.dropout_rate, args.nclass, domain_adaptation=False).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay_rate)

    acc_max, acc_max_step, acc_max_loss, global_step, true_num_test_return = 0, 0, 0, 0, 0
    all_max_train, part_max_train, all_max_test, part_max_test = [], [], [], []

    data_target_iter = cycle(test_loader)

    for epoch in range(args.total_epoch):
        start_time = time.time()
        train_loss = 0
       
        for i, (features, labels) in enumerate(train_loader):
            model.train()

            optimizer.zero_grad()

            s_features, s_labels = features, labels
            s_features = s_features.float().cuda()
            s_labels = s_labels.long().cuda()

         
            outputs, _, part_pro_train1 = model(s_features, adj_pa)
            loss = criterion(outputs, s_labels) + 0.01 * torch.norm(part_pro_train1[1], p=1)

            total_loss = loss 
            train_loss += total_loss
            total_loss.backward()
            optimizer.step()

        # Testing phase (once per epoch)
        test_acc, test_loss, c_matrix, all_max_test = test(model, test_loader, adj_pa, criterion)

        if test_acc > acc_max:
            acc_max = test_acc
            acc_max_loss = test_loss
            acc_max_step = epoch
            torch.save(model.state_dict(), f"{exp_dir}/best_model_subj{t}.pth")

        elapsed = (time.time() - start_time) / 60

        print(
            'subject [{}], Epoch [{}], Time: {:.2f} min, Train_loss: {:.4f}, Test_loss: {:.4f}, Test_acc: {:.2%}, Best_acc: {:.2%}, Best_step: {}'.format(
                t, epoch + 1, elapsed, train_loss, acc_max_loss, test_acc, acc_max, acc_max_step))
        if acc_max == 1:
            break

    return acc_max, c_matrix, all_max_train, all_max_test

def test(model, test_loader, adj_pa, criterion):
    model.eval()
    with torch.no_grad():
        epoch_loss_test = 0
        true_num_test = 0
        N_test = 0
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        all_max_test = None
        for i, (features, labels) in enumerate(test_loader):
            features = features.float().cuda()
            labels = labels.long().cuda()

            outputs, _, all_test = model(features, adj_pa)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            epoch_loss_test += loss.item() * outputs.shape[0]
            true_num_test += (predicted == labels).sum().item()
            N_test += outputs.shape[0]
            all_preds = torch.cat((all_preds.cpu(), predicted.cpu()), dim=0)
            all_labels = torch.cat((all_labels.cpu(), labels.cpu()), dim=0)
            all_max_test = all_test

        test_acc = true_num_test / N_test
        test_loss = epoch_loss_test / N_test
        conf_matrix = confusion_matrix(all_labels.cpu(), all_preds.cpu())

    return test_acc, test_loss, conf_matrix, all_max_test
