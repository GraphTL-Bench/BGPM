import torch
import os.path as osp
import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger
import torch_geometric.transforms as T
from libgptb.data.dataset.abstract_dataset import AbstractDataset
import importlib
from torch_geometric.loader import DataLoader


class PyGGCDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self._load_data()

    def _load_data(self):
        device = torch.device('cuda')
        path = osp.join(os.getcwd(), 'raw_data')

        if self.datasetName in ["IMDB-BINARY","IMDB-MULTI","REDDIT-BINARY", "REDDIT-MULTI-5K" ,"MUTAG", "NCI1", "PROTEINS", "COLLAB", "PTC_MR", "GITHUB_STARGAZER"] and self.config['model']!='JOAO':
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'TUDataset')
            self.dataset = pyg(path, name=self.datasetName, transform=T.NormalizeFeatures())
            self.data = DataLoader(self.dataset, batch_size=128)
        if self.config['model']=='JOAO':
            pyg = getattr(importlib.import_module('aug'), 'TUDataset_aug')
            self.dataset = pyg(path, name=self.datasetName, transform=T.NormalizeFeatures(),aug='minmax')
            self.data = DataLoader(self.dataset, batch_size=128)

        
    
    def get_data(self):
        return self.data
    
    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"input_dim": max(self.dataset.num_features,1)}
    
