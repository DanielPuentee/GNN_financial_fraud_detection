import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import Sequential, Linear, SAGEConv, to_hetero
from torch.nn import ReLU
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


def load_pickle(path, file_name):
    with open(path + file_name, 'rb') as f: return pickle.load(f)

path = "../data/graph_data/"
eda_path = "../data/eda_generated_data/"

training_graph = torch.load(path + 'training_graph.pt')
graph_df = load_pickle(path, 'graph_df.pkl')
df_train = load_pickle(eda_path, 'df_train.pkl')
scaler = load_pickle(eda_path, 'scaler.pkl')

manual_selection = [['ORGANIZATION_TYPE'], ['ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START'], ['NAME_INCOME_TYPE', 'FLAG_OWN_CAR', 'ORGANIZATION_TYPE'], 
                    ['FLAG_OWN_REALTY', 'ORGANIZATION_TYPE'], ['NAME_TYPE_SUITE', 'ORGANIZATION_TYPE'], ['NAME_CONTRACT_TYPE', 'ORGANIZATION_TYPE'], 
                    ['NAME_HOUSING_TYPE', 'NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE'], ['NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS', 'ORGANIZATION_TYPE']]
                    
rename_dict = {'ORGANIZATION_TYPE': 'organization', 
               'ORGANIZATION_TYPE_WEEKDAY_APPR_PROCESS_START': 'organization_weekday',
               'NAME_INCOME_TYPE_FLAG_OWN_CAR_ORGANIZATION_TYPE': 'income_car_organization',
               'FLAG_OWN_REALTY_ORGANIZATION_TYPE': 'realty_organization',
               'NAME_TYPE_SUITE_ORGANIZATION_TYPE': 'suite_organization',
               'NAME_CONTRACT_TYPE_ORGANIZATION_TYPE': 'contract_organization',
               'NAME_HOUSING_TYPE_NAME_EDUCATION_TYPE_ORGANIZATION_TYPE': 'housing_education_organization',
               'NAME_HOUSING_TYPE_NAME_FAMILY_STATUS_ORGANIZATION_TYPE': 'housing_family_organization'}

def join(df, combination_list):

    new_df = pd.DataFrame()
    for i in combination_list:

        df_with_selected_columns = df[list(i)]
        name = '_'.join(i)
        df_with_selected_columns[name] = df_with_selected_columns.apply(lambda x: ' '.join(x), axis=1)
        new_df = pd.concat([new_df, df_with_selected_columns[name]], axis=1)

    return new_df
    
def get_relations(new_df, feature, row):
    filtrado = new_df[feature]
    list_of_index_func = list(filtrado[filtrado == row[feature]].index)
    
    return list_of_index_func

def edge_creation(_user_list_user, new_df, test = None):
    
    if test == None: new_df_copy = new_df.copy()
    else: new_df_copy = new_df.iloc[test:].copy()
        
    for k, row in enumerate(new_df_copy.iterrows()):
        index, value = row

        for x, cols in enumerate(new_df.columns): 
            list_of_index = get_relations(new_df, cols, value)

            lenght_index = len(list_of_index)
            _user_list_user[cols][0] += list_of_index
            _user_list_user[cols][1] += list(np.full(lenght_index, index))

    edges = [torch.tensor([np.array(v[0]), np.array(v[1])], dtype = torch.long) for k, v in _user_list_user.items()]

    return edges

def graph_test(df_test):

    df_test_cleaned = df_test[df_train.columns]
    df_test_cleaned_numeric = df_test_cleaned.select_dtypes(include=['float64', 'int64'])
    df_test_cleaned_numeric_exclude = df_test_cleaned_numeric.drop(['SK_ID_CURR','TARGET'], axis=1)
    df_test_cleaned_numeric_exclude_scaled = scaler.transform(df_test_cleaned_numeric_exclude)

    new_x_values = torch.from_numpy(df_test_cleaned_numeric_exclude_scaled).float()
    training_graph['users'].x = torch.cat((training_graph['users'].x, new_x_values))
    last_val = training_graph['users', 'organization', 'users']['edge_index'][1,-1].item()

    df_test_cleaned.index = range(last_val+1, last_val+1 + len(df_test_cleaned))
    df_test_cleaned_colums_agregated = join(df_test_cleaned, manual_selection)
    df_test_cleaned_colums_agregated.rename(columns=rename_dict, inplace=True)

    graph_df_updated = pd.concat([graph_df, df_test_cleaned_colums_agregated], axis=0)
    _user_list_user, length_test = {i:[[],[]] for i in graph_df_updated.columns}, len(df_test_cleaned_colums_agregated)
    new_edges = edge_creation(_user_list_user, graph_df_updated, -length_test)

    datas_merge = HeteroData()
    for k, v in zip(graph_df_updated.columns, new_edges): datas_merge['users', k, 'users'].edge_index = v
    datas_merge = T.ToUndirected()(datas_merge)
    datas_merge = T.AddSelfLoops()(datas_merge)

    for k, v in enumerate(graph_df_updated.columns):
        previous_relation, new_relations = training_graph['users', v, 'users'].edge_index, datas_merge['users', v, 'users'].edge_index
        new_relation = torch.cat((previous_relation, new_relations), dim=1)
        training_graph['users', v, 'users'].edge_index = new_relation

    test_new_mask = np.array([False] * (last_val+1) + [True] * len(df_test_cleaned_colums_agregated))
    training_graph['users'].test_mask = torch.from_numpy(test_new_mask).bool()

    return training_graph