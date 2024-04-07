import os
import os.path as osp
import dgl
import numpy as np
import scipy.sparse as sp
from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive, load_graphs, save_graphs
from dgl.convert import from_scipy
import torch

class DBLPDataset(DGLDataset):
    url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/{}.npz'

    def __init__(self, root: str, name: str, transform = None, to_undirected: bool = True):
        print("name:{} ".format(name))
        self._name = name.lower()
        self._to_undirected = to_undirected
        assert self._name in ['dblp'], "Dataset name not recognized."
        super().__init__(name=self._name, url=self.url.format(self._name), raw_dir=osp.join(root, self._name, 'raw'), save_dir=osp.join(root, self._name, 'processed'), transform=transform)

    def download(self):
        # 下载数据文件
        file_path = download(self.url.format(self._name), self.raw_dir)
        extract_archive(file_path, self.raw_dir)

    def process(self):
        # 从文件中读取数据
        data = np.load(osp.join(self.raw_dir, f'{self.name}.npz'), allow_pickle=True)

        # 使用提供的键来构造邻接矩阵
        adj_matrix = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=data['adj_shape'])

        # 如果需要将图转换为无向图
        if self._to_undirected:
            adj_matrix = adj_matrix + adj_matrix.T

        # 使用提供的键来构造特征矩阵
        features = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']), shape=data['attr_shape'])
        
        # 转换为密集矩阵格式
        features = features.todense()

        # 读取标签
        labels = data['labels']

        # 将Scipy的稀疏矩阵转换为DGL图
        g = from_scipy(adj_matrix)
        g = dgl.add_self_loop(g)

        # 添加节点特征和标签
        g.ndata['feat'] = torch.FloatTensor(features)
        g.ndata['label'] = torch.LongTensor(labels)

        # 如果有转换函数，则应用它
        if self._transform:
            g = self._transform(g)

        # 保存处理后的图数据
        save_graphs(osp.join(self.save_dir, f'data_{self.name}.bin'), [g])

    def load(self):
        # 加载处理后的图数据
        graphs, _ = load_graphs(osp.join(self.save_dir, f'data_{self._name}.bin'))
        return graphs[0]

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset contains only one graph'
        return self.load()

    def __len__(self):
        return 1
