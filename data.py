# data.py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(data_path, labels_path, batch_size=64, test_ratio=0.2):
    data = np.load(data_path)
    labels = np.load(labels_path)

    data_tensor = torch.from_numpy(data).float()
    labels_tensor = torch.from_numpy(labels).long()

    # 数据转置（[batch_size, 1024, 2] -> [batch_size, 2, 1024]）
    data_tensor = data_tensor.transpose(1, 2)

    # 打乱数据
    indices = torch.randperm(data_tensor.size(0))
    data_shuffled = data_tensor[indices]
    labels_shuffled = labels_tensor[indices]

    # 分割训练集和测试集
    num_samples = data_shuffled.size(0)
    train_size = int((1 - test_ratio) * num_samples)
    train_data, test_data = data_shuffled[:train_size], data_shuffled[train_size:]
    train_labels, test_labels = labels_shuffled[:train_size], labels_shuffled[train_size:]

    # 创建 DataLoader
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
