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

def cleaning_test(df_test):
    
    df_test_cleaned = df_test[df_train.columns]
    df_test_cleaned_numeric = df_test_cleaned.select_dtypes(include=['float64', 'int64'])
    df_test_cleaned_numeric_exclude = df_test_cleaned_numeric.drop(['SK_ID_CURR','TARGET'], axis=1)
    df_test_cleaned_numeric_exclude_scaled = scaler.transform(df_test_cleaned_numeric_exclude)
    
    return df_test_cleaned, df_test_cleaned_numeric_exclude_scaled

def x_values(training_graph, df_test_cleaned_numeric_exclude_scaled):

    new_x_values = torch.from_numpy(df_test_cleaned_numeric_exclude_scaled).float()
    training_graph['users'].x = torch.cat((training_graph['users'].x, new_x_values))
    last_val = training_graph['users', 'organization', 'users']['edge_index'][1,-1].item()
    
    return last_val

def df_test_cleaned_index(df_test_cleaned, last_val):

    df_test_cleaned.index = range(last_val+1, last_val+1 + len(df_test_cleaned))
    df_test_cleaned_colums_agregated = join(df_test_cleaned, manual_selection)
    df_test_cleaned_colums_agregated.rename(columns=rename_dict, inplace=True)
    
    return df_test_cleaned_colums_agregated

def previous_edges_creation(graph_df, last_val, df_test_cleaned_colums_agregated):

    graph_df_updated = pd.concat([graph_df, df_test_cleaned_colums_agregated], axis=0)
    _user_list_user, length_test = {i:[[],[]] for i in graph_df_updated.columns}, len(df_test_cleaned_colums_agregated)
    new_edges = edge_creation(_user_list_user, graph_df_updated, -length_test)
    
    return graph_df_updated, new_edges

def graph_merge(graph_df_updated, new_edges):

    datas_merge = HeteroData()
    for k, v in zip(graph_df_updated.columns, new_edges): datas_merge['users', k, 'users'].edge_index = v
    datas_merge = T.ToUndirected()(datas_merge)
    datas_merge = T.AddSelfLoops()(datas_merge)    
    return datas_merge

def edges_inclusion(graph_df_updated, training_graph, datas_merge):

    for k, v in enumerate(graph_df_updated.columns):
        previous_relation, new_relations = training_graph['users', v, 'users']['edge_index'], datas_merge['users', v, 'users']['edge_index']
        training_graph['users', v, 'users'].edge_index = torch.cat((previous_relation, new_relations), dim=1)
        
    return training_graph

def test_mask_creation(training_graph, length_test, last_val):

    test_mask = np.array([False] * (last_val + 1) + [True] * length_test)
    training_graph['users'].test_mask = torch.from_numpy(test_mask)

    # pass training_graph['users'].valid_mask to numpy array
    valid_mask_numpy = list(training_graph['users'].valid_mask.numpy() ) + [False] * length_test
    training_graph['users'].valid_mask = torch.from_numpy(np.array(valid_mask_numpy))

    train_mask_numpy = list(training_graph['users'].train_mask.numpy() ) + [False] * length_test
    training_graph['users'].train_mask = torch.from_numpy(np.array(train_mask_numpy))

    return training_graph

## NO TEMPORAL DATA TEST TRANSFORMATION ##
def graph_test_no_prev(df_test):

    training_graph = torch.load(path + 'training_graph.pt')

    df_test_cleaned, df_test_cleaned_numeric_exclude_scaled = cleaning_test(df_test)
    last_val = x_values(training_graph, df_test_cleaned_numeric_exclude_scaled)
    df_test_cleaned_colums_agregated = df_test_cleaned_index(df_test_cleaned, last_val)
    graph_df_updated, new_edges = previous_edges_creation(graph_df, last_val, df_test_cleaned_colums_agregated)
    datas_merge = graph_merge(graph_df_updated, new_edges)
    training_graph_modify = edges_inclusion(graph_df_updated, training_graph, datas_merge)
    training_graph_modify = test_mask_creation(training_graph_modify, len(df_test_cleaned_colums_agregated), last_val)

    return training_graph_modify

## TEMPORAL DATA TEST TRANSFORMATION ##
def graph_test_prev(df_test):

    training_graph = torch.load(path + 'training_graph_prev.pt')
    df_previous_graph_all = load_pickle(path, 'df_previous_graph.pkl')
    scaler_previous = load_pickle(path, 'scaler_previous.pkl')
    dict_you_want = load_pickle(path, 'dict_you_want.pkl')
    id_index_dict = load_pickle(path, 'id_index_dict.pkl')

    df_test_cleaned, df_test_cleaned_numeric_exclude_scaled = cleaning_test(df_test)
    last_val = x_values(training_graph, df_test_cleaned_numeric_exclude_scaled)
    df_test_cleaned_colums_agregated = df_test_cleaned_index(df_test_cleaned, last_val)
    graph_df_updated, new_edges = previous_edges_creation(graph_df, last_val, df_test_cleaned_colums_agregated)
    
    df_previous_graph_test = df_previous_graph_all[df_previous_graph_all.SK_ID_CURR.isin(df_test_cleaned.SK_ID_CURR)]
    df_previous_graph_test.index = range(len(dict_you_want), len(dict_you_want) + len(df_previous_graph_test))

    max_dict_value = max(id_index_dict.values())
    id_index_dict_test = {k:v for k, v in zip(df_previous_graph_test.SK_ID_CURR, range(max_dict_value + 1, max_dict_value + len(df_previous_graph_test) + 1))}
    id_index_dict_new = {**id_index_dict, **id_index_dict_test}
    _user_has_previous_test = [[],[]]
    for k, v in id_index_dict_test.items():
        df_graph_prev_filtered = df_previous_graph_test[df_previous_graph_test.SK_ID_CURR == k]
        _user_has_previous_test[0].append(v)
        _user_has_previous_test[1].append(df_graph_prev_filtered.index[0])
    edges_prev_test = torch.tensor([np.array(_user_has_previous_test[0]), np.array(_user_has_previous_test[1])])
    new_x_prev_values = scaler_previous.transform(df_previous_graph_test.drop(['SK_ID_CURR'], axis=1))
    training_graph['previous'].x = torch.cat((training_graph['previous'].x, torch.from_numpy(new_x_prev_values).float()))

    datas_merge = HeteroData()
    for k, v in zip(graph_df_updated.columns, new_edges): datas_merge['users', k, 'users'].edge_index = v
    datas_merge['users', 'has_previous', 'previous'].edge_index = edges_prev_test
    datas_merge = T.ToUndirected()(datas_merge)
    datas_merge = T.AddSelfLoops()(datas_merge)

    training_graph_modify = edges_inclusion(graph_df_updated, training_graph, datas_merge)

    previous_relation, new_relations = training_graph['users', 'has_previous', 'previous'].edge_index, datas_merge['users', 'has_previous', 'previous'].edge_index
    new_relation = torch.cat((previous_relation, new_relations), dim=1)
    training_graph['users', 'has_previous', 'previous'].edge_index = new_relation
    previous_relation_rev, new_relations_rev = training_graph['previous', 'rev_has_previous', 'users'].edge_index, datas_merge['previous', 'rev_has_previous', 'users'].edge_index
    new_relation_rev = torch.cat((previous_relation_rev, new_relations_rev), dim=1)
    training_graph['previous', 'rev_has_previous', 'users'].edge_index = new_relation_rev

    training_graph_modify = test_mask_creation(training_graph_modify, len(df_test_cleaned_colums_agregated), last_val)

    return training_graph_modify
