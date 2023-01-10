import numpy as np
import pickle
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import os
import time
import copy

import torch
import torch_geometric.transforms as T
from torch_geometric.nn import Linear, SAGEConv, to_hetero, GATConv
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import mlflow
import mlflow.pytorch

from customer_inclusion import graph_test_no_prev, graph_test_prev

import warnings
warnings.filterwarnings("ignore")

def save_pickle_file(file_name, file):
    with open(output_path + file_name, 'wb') as f: pickle.dump(file, f)

def load_pickle(path, file_name):
    with open(path + file_name, 'rb') as f: return pickle.load(f)

path = "../data/graph_data/"
eda_path = "../data/eda_generated_data/"
output_path = "../data/models/"

print("Starting connection with MLFlow server...")
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
print("Connection established!")


hyperparams = {
    "aggr_model": "lstm",
    'hidden_channels': 64,
    'hidden_channels2': 32,
    'out_channels': 2,
    'aggr': 'mean',
    'lr': 0.01,
    'epochs': 75
}

class GNN_linear(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, out_channels):
        super().__init__()
        torch.manual_seed(0)
        # self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        # self.conv2 = GATConv((-1, -1), hidden_channels2, add_self_loops=False)
        self.conv1 = SAGEConv((-1, -1), hidden_channels, aggr='mean')
        self.conv2 = SAGEConv((-1, -1), hidden_channels2, aggr='mean')
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

def metrics_return(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return f1, precision, recall, auc

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
    f1, precision, recall, auc = metrics_return(data['users'].y[mask], pred)
    return float(loss), f1, precision, recall, auc

def val(training_graph):

    torch.manual_seed(0)
    model.eval()
    out = model(training_graph.x_dict, training_graph.edge_index_dict)

    mask = training_graph['users'].valid_mask
    pred = out['users'][mask].max(1)[1]
    f1, precision, recall, auc = metrics_return(training_graph['users'].y[mask], pred)
    return f1, precision, recall, auc

def test(y_test, training_graph_new_prev):

    torch.manual_seed(0)
    model.eval()
    out = model(training_graph_new_prev.x_dict, training_graph_new_prev.edge_index_dict)

    mask = training_graph_new_prev['users'].test_mask
    pred = out['users'][mask].max(1)[1]
    f1, precision, recall, auc = metrics_return(y_test, pred)
    return f1, precision, recall, auc

def test_retraining(y_test, training_graph_new_prev):
    
    torch.manual_seed(0)

    model_clone = GNN_linear(hidden_channels=hyperparams['hidden_channels'], hidden_channels2=hyperparams['hidden_channels2'], out_channels=hyperparams['out_channels'])
    model_clone = to_hetero(model_clone, training_graph_new_prev.metadata(), aggr=hyperparams['aggr'])
    model_clone.load_state_dict(model.state_dict())

    optimizer2 = type(optimizer)(model_clone.parameters(), lr=optimizer.defaults['lr'])
    optimizer2.load_state_dict(optimizer.state_dict())

    model_clone.train()
    optimizer2.zero_grad()
    out = model_clone(training_graph_new_prev.x_dict, training_graph_new_prev.edge_index_dict)
    mask = training_graph_new_prev['users'].train_mask
    loss = criterion(out['users'][mask], training_graph_new_prev['users'].y[mask])
    loss.backward()
    optimizer2.step()

    mask = training_graph_new_prev['users'].test_mask
    pred = out['users'][mask].max(1)[1]
    f1, precision, recall, auc = metrics_return(y_test, pred)

    return f1, precision, recall, auc

def adding_scalars():
            
        writer.add_scalar('Loss/train', loss, epoch)
        
        writer.add_scalar('F1/train', f1_train, epoch)
        writer.add_scalar('Precision/train', precision_train, epoch)
        writer.add_scalar('Recall/train', recall_train, epoch)
        writer.add_scalar('AUC/train', auc_train, epoch)

        writer.add_scalar('F1/val', f1_val, epoch)
        writer.add_scalar('Precision/val', precision_val, epoch)
        writer.add_scalar('Recall/val', recall_val, epoch)
        writer.add_scalar('AUC/val', auc_val, epoch)

        writer.add_scalar('F1/test', f1_test, epoch)
        writer.add_scalar('Precision/test', precision_test, epoch)
        writer.add_scalar('Recall/test', recall_test, epoch)
        writer.add_scalar('AUC/test', auc_test, epoch)
        

if __name__ == "__main__":

    writer = SummaryWriter()
    
    training_graph = torch.load(path + 'training_graph_prev.pt')
    df_test = load_pickle(eda_path, "df_test.pkl")

    training_graph_new_prev = graph_test_prev(df_test.iloc[:1000])
    y_test = df_test.iloc[:1000].TARGET
    y_test_torch = torch.from_numpy(y_test.values).long()
    training_graph_new_prev['users'].y = torch.cat((training_graph['users'].y, y_test_torch), 0)
    


    torch.manual_seed(0)
    model = GNN_linear(hidden_channels=hyperparams['hidden_channels'], hidden_channels2=hyperparams['hidden_channels2'], out_channels=hyperparams['out_channels'])
    model = to_hetero(model, training_graph.metadata(), aggr=hyperparams['aggr'])

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
    y = np.array(training_graph['users'].y)
    weights = [1 - (len(y[y == 0])/len(y)), 1 - (len(y[y == 1])/len(y))]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))

    mlflow.set_experiment("Pytorch Model - No temporal data")
    
    print('Training...')
    start = time.time()

    torch.manual_seed(0)
    for epoch in range(1, hyperparams['epochs'] + 1):
        
        torch.manual_seed(0)
        loss, f1_train, precision_train, recall_train, auc_train = train(training_graph)
        f1_val, precision_val, recall_val, auc_val = val(training_graph)
        f1_test, precision_test, recall_test, auc_test = test(y_test, training_graph_new_prev)
        # f1_test, precision_test, recall_test, auc_test = test_retraining(y_test, training_graph_new_prev)
        

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, | F1 Train: {f1_train:.3f}, F1 Val: {f1_val:.3f}, F1 Test: {f1_test:.3f}, | Precision Train: {precision_train:.3f}, Precision Val: {precision_val:.3f}, Precision Test: {precision_test:.3f}, | Recall Train: {recall_train:.3f}, Recall Val: {recall_val:.3f}, Recall Test: {recall_test:.3f}, | AUC Train: {auc_train:.3f}, AUC Val: {auc_val:.3f}, AUC Test: {auc_test:.3f}')
        metrics = {'loss': loss, 'f1_train': f1_train, 'precision_train': precision_train, 'recall_train': recall_train, 'auc_train': auc_train, 'f1_val': f1_val, 'precision_val': precision_val, 'recall_val': recall_val, 'auc_val': auc_val, 'f1_test': f1_test, 'precision_test': precision_test, 'recall_test': recall_test, 'auc_test': auc_test}
        mlflow.log_metrics(metrics)
        minutes = (time.time() - start)/60
        print(f'Time: {minutes:.3f} min')

        adding_scalars()

        

    print('¡¡¡Training finished!!!')
    mlflow.log_params(hyperparams)
    mlflow.set_tag("Model name", "sage_aggr_mean_lstm_64_75")
    mlflow.pytorch.log_model(model, artifact_path = "sage_aggr_mean_lstm_64_75")
    torch.save(model, output_path + 'sage_aggr_mean_lstm_64_75.pt')
    writer.close()