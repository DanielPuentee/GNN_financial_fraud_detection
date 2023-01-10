import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import Sequential, Linear, SAGEConv, to_hetero
from torch.nn import ReLU
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def save_pickle_file(file_name, file):
    with open(output_path + file_name, 'wb') as f: pickle.dump(file, f)

def load_pickle(path, file_name):
    with open(path + file_name, 'rb') as f: return pickle.load(f)

path = "../data/graph_data/"
eda_path = "../data/eda_generated_data/"
output_path = "../data/models/"

training_graph = torch.load(path + 'training_graph_prev.pt')


class GNN_linear(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, out_channels):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels2)
        self.lin = Linear(hidden_channels2, out_channels)

    def forward(self, x, edge_index):
        torch.manual_seed(0)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.lin(x)
        x = F.sigmoid(x)
        return x

torch.manual_seed(0)
model = GNN_linear(hidden_channels=16, hidden_channels2=8, out_channels=2)
model = to_hetero(model, training_graph.metadata(), aggr='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

y = np.array(training_graph['users'].y)
weights = [1 - (len(y[y == 0])/len(y)), 1 - (len(y[y == 1])/len(y))]
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))

torch.manual_seed(0)
def train(data):

    torch.manual_seed(0)
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['users'].train_mask
    
    loss = criterion(out['users'][mask], data['users'].y[mask])

    loss.backward()
    optimizer.step()

    pred = out['users'][mask].max(1)[1]
    f1 = f1_score(data['users'].y[mask], pred)
    precion = precision_score(data['users'].y[mask], pred)
    recall = recall_score(data['users'].y[mask], pred)
    auc = roc_auc_score(data['users'].y[mask], pred)
    
    return float(loss), f1, precion, recall, auc

def test(training_graph):

    torch.manual_seed(0)
    model.eval()
    out = model(training_graph.x_dict, training_graph.edge_index_dict)

    mask = training_graph['users'].valid_mask
    pred = out['users'][mask].max(1)[1]
    f1 = f1_score(training_graph['users'].y[mask], pred)
    precision = precision_score(training_graph['users'].y[mask], pred)
    recall = recall_score(training_graph['users'].y[mask], pred)
    auc = roc_auc_score(training_graph['users'].y[mask], pred)

    return f1, precision, recall, auc

stats = pd.DataFrame(columns=['epoch', 'loss', 'f1_train', 'precision_train', 'recall_train', 'auc_train', 'f1_val', 'precision_val', 'recall_val', 'auc_val'])
print('Training...')
torch.manual_seed(0)
for epoch in range(1, 20+1):

    torch.manual_seed(0)
    loss, f1_train, precision_train, recall_train, auc_train = train(training_graph)
    f1_val, precision_val, recall_val, auc_val = test(training_graph)

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, F1 Train: {f1_train:.3f}, F1 Val: {f1_val:.3f}, Precision Train: {precision_train:.3f}, Precision Val: {precision_val:.3f}, Recall Train: {recall_train:.3f}, Recall Val: {recall_val:.3f}, AUC Train: {auc_train:.3f}, AUC Val: {auc_val:.3f}')
    stats = stats.append({'epoch': epoch, 'loss': loss, 'f1_train': f1_train, 'precision_train': precision_train, 'recall_train': recall_train, 'auc_train': auc_train, 'f1_val': f1_val, 'precision_val': precision_val, 'recall_val': recall_val, 'auc_val': auc_val}, ignore_index=True)

print('¡¡¡Training finished!!!')
torch.save(model, output_path + 'sage_heterogen_model_prev.pt')
save_pickle_file('sage_heterogen_model_stats_prev.pkl', stats)