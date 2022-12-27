import torch 
import numpy as np
from torch.utils.data import Dataset

class DCRNNDataset(Dataset):
    def __init__(self, data):
        x = torch.tensor(data['x']).float().unsqueeze(-1)
        y = torch.tensor(data['y']).float().unsqueeze(-1)

        if data['tod'] is not None: 
            x = torch.cat([x, torch.tensor(data['tod'][0]).unsqueeze(-2).repeat(1, 1, x.size(-2), 1)], dim=-1)
            y = torch.cat([y, torch.tensor(data['tod'][1]).unsqueeze(-2).repeat(1, 1, x.size(-2), 1)], dim=-1)

        if data['dow'] is not None: 
            x = torch.cat([x, torch.tensor(data['dow'][0]).unsqueeze(-2).repeat(1, 1, x.size(-2), 1)], dim=-1)
            y = torch.cat([y, torch.tensor(data['dow'][1]).unsqueeze(-2).repeat(1, 1, x.size(-2), 1)], dim=-1)
        
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()
    
    def __len__(self):
        return self.y.size(0)

class STDENDataset(Dataset):
    def __init__(self, data):
        self.x = data['x'] 
        self.y = data['y']

    def __getitem__(self, index):
        return torch.tensor(self.x[index]).float(), torch.tensor(self.y[index]).float()
    
    def __len__(self):
        return self.y.shape[0]

class GMANDataset(Dataset):
    def __init__(self, data):
        self.x = data['x'] 
        self.y = data['y']
        self.te = np.concatenate([np.concatenate([data['tod'][0],data['tod'][1]], axis=1), np.concatenate([data['dow'][0],data['dow'][1]], axis=1)], axis=-1)

    def __getitem__(self, index):
        return torch.tensor(self.x[index]).float(), torch.tensor(self.te[index]).float(), torch.tensor(self.y[index]).float()
    
    def __len__(self):
        return self.y.shape[0]

class STGCNDataset(Dataset):
    def __init__(self, data):
        x = torch.tensor(data['x']).float().unsqueeze(1)
        y = torch.tensor(data['y']).float()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()
    
    def __len__(self):
        return self.y.size(0)


class WaveNetDataset(Dataset):
    def __init__(self, data):
        x = torch.tensor(data['x']).float().unsqueeze(-1)
        y = torch.tensor(data['y']).float().unsqueeze(-1)

        if data['tod'] is not None: 
            x = torch.cat([x, torch.tensor(data['tod'][0]).unsqueeze(-2).repeat(1, 1, x.size(-2), 1)], dim=-1)
            y = torch.cat([y, torch.tensor(data['tod'][1]).unsqueeze(-2).repeat(1, 1, x.size(-2), 1)], dim=-1)

        if data['dow'] is not None: 
            x = torch.cat([x, torch.tensor(data['dow'][0]).unsqueeze(-2).repeat(1, 1, x.size(-2), 1)], dim=-1)
            y = torch.cat([y, torch.tensor(data['dow'][1]).unsqueeze(-2).repeat(1, 1, x.size(-2), 1)], dim=-1)
        
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()
    
    def __len__(self):
        return self.y.size(0)


class STGODEDataset(Dataset):
    def __init__(self, data, his_length, pred_length):
        self.data = data.astype(np.float32)
        self.his_length = his_length
        self.pred_length = pred_length
    
    def __getitem__(self, index):
        x = self.data[index: index + self.his_length]
        y = self.data[index + self.his_length: index + self.his_length + self.pred_length][:, :, 0]
        return torch.Tensor(x).permute(1, 0, 2), torch.Tensor(y).permute(1, 0)

    def __len__(self):
        return self.data.shape[0] - self.his_length - self.pred_length + 1
