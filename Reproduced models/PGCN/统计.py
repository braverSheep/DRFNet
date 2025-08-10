
from fvcore.nn import FlopCountAnalysis, flop_count_table
import model_PGCN as ss
import torch
import time
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义格式化函数

def format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f} B"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{n/1e3:.2f} K"
    return str(n)


def format_flops(n: int) -> str:
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"{n/1e12:.2f} TFLOPs"
    if n >= 1e9:
        return f"{n/1e9:.2f} GFLOPs"
    if n >= 1e6:
        return f"{n/1e6:.2f} MFLOPs"
    if n >= 1e3:
        return f"{n/1e3:.2f} KFLOPs"
    return f"{n} FLOPs"


import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:3", help="gpu device")
    # parser.add_argument('--augment_feature_attention', type=int, default=20,
    #                     help='Feature augmentation matrix for attention.')
    parser.add_argument("--log_dir", type=str, default=os.path.join("..", "logs", ), help="log file dir")
    parser.add_argument('--out_feature', type=int, default=20, help='Output feature for GCN.')
    parser.add_argument('--seed', type=int, default=222, help='Random seed.')  # 42
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=20, help='early stopping param')
    # hyperparameter
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    # parser.add_argument('--alpha', type=float, default=0.005, help='Attention reconciliation hyperparameters')  # 5e-4
    parser.add_argument('--beta', type=float, default=5e-5, help='update laplacian matrix')  # 5e-4
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')

    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate .')
    # parser.add_argument('--leakyrelu', type=float, default=0.1, help='leaky relu.')
    # pri-defined dataset
    parser.add_argument("--dataset", type=str, default="SEED4", help="dataset: SEED, SEED4, SEED5, MPED, bcic ")
    parser.add_argument("--session", type=str, default="2", help="")
    parser.add_argument("--mode", type=str, default="dependent", help="dependent, independent or transfer")
    parser.add_argument("--checkpoint", type=str, default=None, help="store current subject's checkpoint")
    parser.add_argument("--module", type=str, default="", help="Store which modules are used in this run")

    # 定义数据集相关的部分参数
    args = parser.parse_args()
    if args.dataset == 'SEED':
        pass
    elif args.dataset == 'SEED4':
        parser.add_argument("--in_feature", type=int, default=5, help="")
        parser.add_argument("--n_class", type=int, default=4, help="")
        parser.add_argument("--epsilon", type=float, default=0.05, help="")
        parser.add_argument("--datapath", type=str, default="../npy_data/seed4/", help="")
    elif args.dataset == 'SEED5':
        pass
    elif args.dataset == 'MPED':
        pass
    else:
        raise ValueError("Wrong dataset!")

    return parser.parse_args()



def getdata():
    from torch_geometric.data import Data

    # 假设 num_nodes=62, batch_size=8, in_features=64
    batch_size, num_nodes, in_features = 1, 62, 5

    # 1) x: (8*62, 64)
    x = torch.randn(batch_size * num_nodes, in_features)

    # 2) edge_index: [2, E]  例如完全图 E = 62*62
    row = torch.arange(num_nodes).repeat(num_nodes)
    col = torch.arange(num_nodes).unsqueeze(1).repeat(1, num_nodes).flatten()
    edge_index = torch.stack([row, col], dim=0)

    # 3) y: (8,)
    y = torch.randint(0, 4, (batch_size,))

    # 4) batch: (8*62,)
    batch = torch.arange(batch_size).repeat_interleave(num_nodes)

    data = Data(x=x, edge_index=edge_index, y=y, batch=batch)

    return data

def measure_inference_time(
    model,
    input_size,
    device,
    n_runs: int = 100
) -> float:
    """
    Measure average inference time over n_runs.
    """
    model = model.to(device)
    model.eval()

    data = getdata()
    # create dummy input
    sample = torch.randn(32,62,5).to(device)

    # warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    return sum(times) / len(times)



data = getdata()

sample = {
        'x': data.x.to(device),
        'y': data.y.to(device),
        'batch': data.batch.to(device),
        'edge_index': data.edge_index.to(device),
    }

args = parse_args()
adj = torch.randn([62,62]).flatten()

from node_location import convert_dis_m, get_ini_dis_m, return_coordinates
from torch.nn.parameter import Parameter
# 返回节点的绝对坐标
coordinate_matrix = torch.FloatTensor(return_coordinates()).to(device)
# 局部视野的邻接矩阵
adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), 9))).to(device)

model = ss.PGCN(args, adj_matrix, coordinate_matrix,).to(device)
model.eval() 



# 1. 打印每个子模块（layer）的参数量
print("Parameters per layer:")
for name, module in model.named_modules():
    # 跳过没有可训练参数的模块
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if params > 0:
        print(f"{name:40s} ─ {format_params(params)}")


sample = torch.randn(32,62,5).to(device)
# 统计参数和 FLOPs
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
flops = FlopCountAnalysis(model, sample)
total_flops = flops.total()

# 按论文格式打印结果
print(f"Total Params: {format_params(total_params)}")
print(f"Total FLOPs: {format_flops(total_flops)}")


# 统计推理时间

avg_time = measure_inference_time(model, 1, device, 100)
print(f"Average inference time over {100} runs: {avg_time:.6f} seconds")
