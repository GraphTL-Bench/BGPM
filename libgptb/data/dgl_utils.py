import importlib
import numpy as np
from torch.utils.data import DataLoader
import copy

import os
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset



def get_dataset(config):
    """
    according the config['dataset_class'] to create the dataset

    Args:
        config(ConfigParser): config

    Returns:
        AbstractDataset: the loaded dataset
    """
    try:
        return getattr(importlib.import_module('libgptb.data.dataset'),
                       config['dataset_class'])(config)
    except AttributeError:
        try:
            return getattr(importlib.import_module('libgptb.data.dataset.dataset_subclass'),
                           config['dataset_class'])(config)
        except AttributeError:
            raise AttributeError('dataset_class is not found')


def load_pcg(data_name, split_name, train_percent, path="raw_data/"):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()
    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    edges = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edges_undir.csv"),
                                       dtype=np.dtype(float), delimiter=','))
    edge_index = torch.transpose(edges, 0, 1).long()
    edge_weight = torch.sum(labels[edge_index[0, :]] * labels[edge_index[1, :]], 1).float()

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True
    #print(test_mask[:5])

    y = labels.clone().detach().float()
    y[val_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])
    y[test_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])

    return features, edge_index, edge_weight, y, train_mask, val_mask, test_mask



def load_humloc(data_name="HumanGo", path="raw_data/"):
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    print("edge_index shape:", edge_index.shape)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))
    print("edge_index_other_half shape:", edge_index_other_half.shape)
    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    edge_weight = torch.sum(labels[edge_index[0, :]] * labels[edge_index[1, :]], 1).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()
    print("features shape:", features.shape)
    file_path = os.path.join(path, data_name, "split.pt")
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    y = labels.clone().detach().float()
    y[val_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])
    y[test_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])
    

    return features, edge_index, edge_weight, y, train_mask, val_mask, test_mask



def load_eukloc(data_name="EukaryoteGo", path="raw_data/"):
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    print("edge_index shape:", edge_index.shape)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))
    print("edge_index_other_half shape:", edge_index_other_half.shape)
    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    edge_weight = torch.sum(labels[edge_index[0, :]] * labels[edge_index[1, :]], 1).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()
    print("features shape:", features.shape)
    file_path = os.path.join(path, data_name, "split.pt")
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    y = labels.clone().detach().float()
    y[val_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])
    y[test_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])
    

    return features, edge_index, edge_weight, y, train_mask, val_mask, test_mask

class EuklocDataset(DGLDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(EuklocDataset, self).__init__(name='Eukloc',
                                          url=None,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)


    def process(self):
        features, edge_index, edge_weight, y, train_mask, val_mask, test_mask = load_eukloc()

        g = dgl.graph((edge_index[0], edge_index[1]),num_nodes=features.shape[0])
        print("Features shape:", features.shape)
        print("Number of nodes:", g.number_of_nodes())
        g.ndata['feat'] = features
        g.edata['weight'] = edge_weight
        g.ndata['label'] = y
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        self.data = g
        self.save()

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1
        
class HumlocDataset(DGLDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(HumlocDataset, self).__init__(name='Humloc',
                                          url=None,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)


    def process(self):
        features, edge_index, edge_weight, y, train_mask, val_mask, test_mask = load_humloc()

        g = dgl.graph((edge_index[0], edge_index[1]),num_nodes=features.shape[0])
        print("Features shape:", features.shape)
        print("Number of nodes:", g.number_of_nodes())
        g.ndata['feat'] = features
        g.edata['weight'] = edge_weight
        g.ndata['label'] = y
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        self.data = g
        self.save()

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1

class PcgDataset(DGLDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(PcgDataset, self).__init__(name='pcg',
                                          url=None,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)


    def process(self):
        features, edge_index, edge_weight, y, train_mask, val_mask, test_mask = load_pcg('pcg_removed_isolated_nodes','split_0.pt',0.6)

        g = dgl.graph((edge_index[0], edge_index[1]),num_nodes=features.shape[0])
        print("Features shape:", features.shape)
        print("Number of nodes:", g.number_of_nodes())
        g.ndata['feat'] = features
        g.edata['weight'] = edge_weight
        g.ndata['label'] = y
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        self.data = g
        self.save()

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1
