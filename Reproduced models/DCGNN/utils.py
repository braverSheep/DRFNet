import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio

def normalize_A(A, symmetry=False):
    #print(A.ndim)
    if symmetry and A.ndim==2:
        A=A+A.T
        d = torch.sum(A, -1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d1=torch.sum(A,-1)
        d1=1 / torch.sqrt(d1 + 1e-10)
        D1=torch.diag_embed(d1)
        d2=torch.sum(A,-2)
        d2=1 / torch.sqrt(d2 + 1e-10)
        D2=torch.diag_embed(d2)
        L = torch.matmul(torch.matmul(D1, A), D2)
    return L

def generate_cheby_adj(A, K):
    support = []
    for i in range(K):
        if i == 0:
            # support.append(torch.eye(A.shape[1]).cuda())
            support.append(torch.eye(A.shape[1]).cuda())
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support

def generate_R(nfeat, part):
    mask = torch.zeros((nfeat, 16)).cuda()
    for i in range(16):
        for j in part[i]:
            mask[j, i] = 1.0
    return mask



def loss_I(part_pro):
    result=-torch.mul(part_pro,torch.log(part_pro+1e-10))
    result = torch.sum(result)/ part_pro.shape[0]
    return result

def loss_J(part_pro):
    part_pro=F.normalize(part_pro, p=2, dim=1)
    part_pro = torch.matmul(part_pro, part_pro.transpose(1, 2))
    I = torch.ones(part_pro.shape[1]).cuda() - torch.eye(part_pro.shape[1]).cuda()
    result = part_pro * I
    result = torch.sum(result) / part_pro.shape[0]
    return result

def generate_mask(nfeat, part):
    mask = torch.zeros((nfeat, nfeat)).cuda()
    for i in range(nfeat):
        for j in range(nfeat):
            for m in range(len(part)):
                if np.isin(i, part[m]) and np.isin(j, part[m]):
                    mask[i, j] = 1
    return mask

def preprocessX(xtrain,xtest):
    nsam, ncha, nfea = xtrain.shape
    mean_data = np.mean(xtrain.reshape(nsam * ncha, nfea), axis=0)  # mean data for each band
    std_data = np.std(xtrain.reshape(nsam * ncha, nfea), axis=0)
    xtrain = (xtrain - mean_data) / std_data
    xtest = (xtest - mean_data) / std_data
    return xtrain,xtest

def contrastive_loss(contrastive_output, temperature=0.5):

    contrastive_output = F.normalize(contrastive_output, p=2, dim=1)
    similarity_matrix = torch.matmul(contrastive_output, contrastive_output.T)
    similarity_matrix /= temperature


    batch_size = contrastive_output.size(0)
    labels = torch.arange(batch_size).cuda()
    mask = torch.eye(batch_size, dtype=torch.bool).cuda()

    positives = similarity_matrix[mask].view(batch_size, -1)
    negatives = similarity_matrix[~mask].view(batch_size, -1)


    labels = torch.zeros(batch_size).long().cuda()
    loss = F.cross_entropy(torch.cat([positives, negatives], dim=1), labels)

    return loss


def label_smoothing_loss(preds, targets, smoothing=0.1):
    n_classes = preds.size(1)
    one_hot = torch.zeros_like(preds).scatter(1, targets.unsqueeze(1), 1)
    smoothed_labels = one_hot * (1 - smoothing) + (smoothing / n_classes)
    log_probs = F.log_softmax(preds, dim=1)
    # loss = -torch.sum(smoothed_labels * log_probs, dim=1).mean()
    loss = F.kl_div(log_probs, smoothed_labels, reduction='batchmean')
    return loss

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def CDAN_no_random(input_list, ad_net, entropy=None, coeff=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]


    combined_feature = torch.cat((feature, softmax_output), dim=1)


    ad_out = ad_net(combined_feature,coeff)


    batch_size = softmax_output.size(0) // 2
    dc_target = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).float().cuda()


    return nn.BCELoss()(ad_out, dc_target.unsqueeze(1))

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)),coeff)
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)),0)
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)
