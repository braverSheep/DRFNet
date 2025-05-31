# 示例：leiden_on_aggregated_adj.py
# pip install python-igraph leidenalg
import torch
import numpy as np
import igraph as ig
import leidenalg as la
from collections import Counter
import random


seed=12
random.seed(seed)
np.random.seed(seed)

# —— 1. 加载已聚合的邻接矩阵 —— 
# （假设你运行过 aggregate_edge_weights()，并保存为 aggregated_adj.pt）
agg_adj_path = "aggregated_adj.pt"
agg_adj = torch.load(agg_adj_path, map_location="cpu")  # Tensor [62×62]
A = agg_adj.numpy()


# —— 2. 对称化 & （可选）阈值化 —— 
# 保证无向图，对称化
A = (A + A.T) / 2

print(A)
# 可选：只保留 top 25% 强度的边，去除噪声
thresh = np.percentile(A[np.triu_indices_from(A, k=1)], 90) # 60
A_filtered = np.where(A >= thresh, A, 0.0)

# —— 3. 构建 igraph 图 —— 
# igraph 要求邻接矩阵列表形式，模式为无向加权图
G = ig.Graph.Weighted_Adjacency(
    A_filtered.tolist(),
    mode=ig.ADJ_UNDIRECTED,
    attr="weight",
    loops=False
)

# —— 4. Leiden 社区发现 —— 
# RBConfigurationVertexPartition 对应模块度最大化；调节 resolution_parameter 控制区数
partition = la.find_partition(
    G,
    la.RBConfigurationVertexPartition,
    weights="weight",
    resolution_parameter=1.0,
    seed=seed
)

# —— 5. 输出每个电极的“功能区”标签 —— 
labels = np.array(partition.membership)  
print("电极索引 → 社区标签：", labels)
print(f"共检测到 {len(set(labels))} 个簇（脑区）")

# —— 6. 统计每个簇的节点数 ——
cluster_counts = Counter(labels)
print("各簇电极数量：")
for cluster_id in sorted(cluster_counts):
    print(f"  簇 {cluster_id}: {cluster_counts[cluster_id]} 个电极")

# —— 6. 分配（社区标签数组） —— 

class Groups:
    def __init__(self, labels):
        # —— 将你的社区标签粘到这里 —— 
        # labels = [
        #     0, 3, 3, 0, 3, 1, 3, 3, 0, 0, 4, 4, 4, 2, 2, 2, 2, 0, 1, 0, 0, 4, 2, 2, 2,
        #     2, 3, 3, 2, 4, 2, 2, 1, 1, 2, 3, 3, 0, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2,
        #     1, 2, 1, 4, 1, 0, 1, 0, 1, 1, 4, 0, 0
        # ]  # len = 62

        # 初始化 7 个组
        for k in range(1, 8):
            setattr(self, f"channel_group{k}", [])

        # 1–62 号电极，label==g-1 的归入 channel_groupg
        for idx, lab in enumerate(labels):
            group_id = lab + 1           # lab 0 → group1, lab1 → group2, … lab4 → group5
            if 1 <= group_id <= 7:
                getattr(self, f"channel_group{group_id}").append(idx + 1)

        # 打印检查
        for k in range(1, 8):
            grp = getattr(self, f"channel_group{k}")
            print(f"channel_group{k} ({len(grp)}): {grp}")


groups = Groups(labels)


'''
我们的结果
[[ 1.4378852   0.13790043 -0.02553363 ... -0.01776607 -0.14070813
  -0.11622977]
 [ 0.13790043  1.7231877   0.19255313 ... -0.12085487 -0.0175439
   0.01661648]
 [-0.02553363  0.19255313  1.3151008  ... -0.08243787  0.03186987
   0.10560261]
 ...
 [-0.01776607 -0.12085487 -0.08243787 ...  1.5530381   0.09364974
   0.05396828]
 [-0.14070813 -0.0175439   0.03186987 ...  0.09364974  1.178565
   0.20335823]
 [-0.11622977  0.01661648  0.10560261 ...  0.05396828  0.20335823
   1.1143231 ]]
电极索引 → 社区标签： [0 3 3 0 3 1 3 3 0 0 4 4 4 2 2 2 2 0 1 0 0 4 2 2 2 2 3 3 2 4 2 2 1 1 2 3 3
 0 2 1 1 1 1 1 0 0 0 0 2 1 2 1 4 1 0 1 0 1 1 4 0 0]
共检测到 5 个簇（脑区）
各簇电极数量：
  簇 0: 16 个电极
  簇 1: 15 个电极
  簇 2: 15 个电极
  簇 3: 9 个电极
  簇 4: 7 个电极
channel_group1 (16): [1, 4, 9, 10, 18, 20, 21, 38, 45, 46, 47, 48, 55, 57, 61, 62]
channel_group2 (15): [6, 19, 33, 34, 40, 41, 42, 43, 44, 50, 52, 54, 56, 58, 59]
channel_group3 (15): [14, 15, 16, 17, 23, 24, 25, 26, 29, 31, 32, 35, 39, 49, 51]
channel_group4 (9): [2, 3, 5, 7, 8, 27, 28, 36, 37]
channel_group5 (7): [11, 12, 13, 22, 30, 53, 60]
channel_group6 (0): []
channel_group7 (0): []
'''