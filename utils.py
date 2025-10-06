import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch

class TestbedDataset(InMemoryDataset):
    def __init__(self, root, dataset, xd=None, xt=None, y=None, transform=None, pre_transform=None, smile_graph=None):
        self.dataset = dataset
        self.xd = xd
        self.xt = xt
        self.y = y
        self.smile_graph = smile_graph
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        if os.path.isfile(self.processed_paths[0]):
            print(f"✅ Pre-processed data found: {self.processed_paths[0]}, loading...")
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            if xd is None or xt is None or y is None or smile_graph is None:
                raise ValueError("No raw data provided and no preprocessed file found!")
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return [self.dataset + '.pt']

    def process(self):
        data_list = []
        for i, smiles in enumerate(self.xd):
            target = self.xt[i]
            labels = self.y[i]
            c_size, features, edge_index = self.smile_graph[smiles]
            graph_data = DATA.Data(
                x=torch.tensor(features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                y=torch.tensor([labels], dtype=torch.float)
            )
            graph_data.target = torch.tensor([target], dtype=torch.long)
            graph_data.c_size = torch.tensor([c_size], dtype=torch.long)
            data_list.append(graph_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# ===== Metrics =====
def rmse(y, f): return sqrt(((y - f) ** 2).mean(axis=0))
def mse(y, f): return ((y - f) ** 2).mean(axis=0)
def pearson(y, f): return np.corrcoef(y, f)[0, 1]
def spearman(y, f): return stats.spearmanr(y, f)[0]
def ci(y, f):
    ind = np.argsort(y)
    y, f = y[ind], f[ind]
    S = z = 0.0
    for i in range(len(y)-1, 0, -1):
        for j in range(i-1, -1, -1):
            if y[i] > y[j]:
                z += 1
                u = f[i] - f[j]
                if u > 0: S += 1
                elif u == 0: S += 0.5
    return S / z
