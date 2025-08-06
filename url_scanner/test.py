import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader 
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("url_scanner/save_urlScanner/transformer_ver_1.0-2025-08-06.pth").to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


url = "https://pvz-fusion.fandom.com/wiki/PvZ:_Fusion_Wiki"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.eval()
encoding =tokenizer(url,max_length=64,padding='max_length',truncation=True,return_tensors='pt')
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device) 

with torch.no_grad():
        output = model(input_ids, attention_mask).squeeze(1)
        prob = torch.sigmoid(output).item()
        pred_label = "malicious" if prob > 0.5 else "safe"
    