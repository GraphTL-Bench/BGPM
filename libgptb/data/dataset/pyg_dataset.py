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

class PyGDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.datasetName = self.config.get('dataset', '')
        self._load_data()

    def _load_data(self):
        device = torch.device('cuda')
        path = osp.join(os.getcwd(), 'raw_data')

        if self.datasetName in ["Cora", "CiteSeer", "PubMed"]:
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'Planetoid')
        if self.datasetName in ["Computers", "Photo"]:
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'Amazon')
        if self.datasetName in ["CS", "Physics"]:
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'Coauthor')
            
        if self.datasetName in ["Yelp"]:
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'Yelp')
        if self.datasetName in ["AmazonProducts"]:
            pyg = getattr(importlib.import_module('torch_geometric.datasets'), 'AmazonProducts')
            print(f"{self.datasetName} loaded")
            
        if self.datasetName not in ["Yelp","AmazonProducts","PPI"]:
            self.dataset = pyg(path, name=self.datasetName, transform=T.NormalizeFeatures())

        if self.datasetName in ["Yelp"]:
            self.dataset = pyg(path, transform=T.NormalizeFeatures())
        if self.datasetName in ["AmazonProducts"]:
            path = osp.join(os.getcwd(), 'raw_data', 'AP')
            self.dataset = pyg(path, transform=T.NormalizeFeatures())
            print(f"{self.datasetName} loaded")

        if self.datasetName in ["PPI"]:
            PPI = getattr(importlib.import_module('torch_geometric.datasets'), 'PPI')        
            self.train_dataset = PPI(path, split='train')
            val_dataset = PPI(path, split='val')
            test_dataset = PPI(path, split='test')
            self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
            self.test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        # self.data = 
        
    
    def get_data(self):
        # loader = DataLoader(self.dataset,batch_size=1)
        device = torch.device('cuda')
        if self.datasetName not in ["PPI"]:
            return self.dataset[0].to(device)
        else:
            return self.train_loader,self.val_loader,self.test_loader
    
    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        if self.datasetName not in ["PPI"]:
            return {"input_dim": self.dataset.num_features}
        else:
            return {"input_dim": self.train_dataset.num_features}
    
