import numpy as np
import torch.nn as nn 
import torch 

class MaskedMSE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        mask = mask.to(torch.float32).to(preds.device)
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.pow(preds - labels, 2)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

class MaskedMAE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        mask = mask.to(torch.float32).to(preds.device)
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds - labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

class MaskedRMSE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        mask = mask.to(torch.float32).to(preds.device)
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.pow(preds - labels, 2)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.sqrt(torch.mean(loss))

class MaskedMAPE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        mask = mask.to(torch.float32).to(preds.device)
        mask /= torch.mean(mask)
        loss = torch.abs(torch.divide(torch.subtract(preds, labels), labels))
        loss = torch.nan_to_num(loss * mask)
        return torch.mean(loss)