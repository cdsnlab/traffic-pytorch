import torch 
from torch.utils.data import Dataset

class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DRCNNDataset(Dataset):
    def __init__(self, x, y):
        self.x = x 
        self.y = y 

    def __getitem__(self, index):
        return torch.tensor(self.x[index]).float(), torch.tensor(self.y[index]).float()
    
    def __len__(self):
        return self.y.shape[0]