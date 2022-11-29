import pickle 
import numpy as np
import os
import torch
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros((num_sample, num_his, dims))
    y = np.zeros((num_sample, num_pred, dims))
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def generate_train_val_test(dataset_dir, dataset_name, train_ratio, test_ratio, add_dayofweek=False, add_timeofday=False):
    df = pd.read_hdf(os.path.join(dataset_dir, dataset_name)+".h5")
    traffic = df.values

    time = pd.DatetimeIndex(df.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1)).numpy()
    timeofday = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1)).numpy()

    num_step = df.shape[0]
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = traffic[:train_steps]
    val = traffic[train_steps:train_steps+val_steps]
    test = traffic[-test_steps:]

    np.save("{}{}_train_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), train)
    np.save("{}{}_val_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), val)
    np.save("{}{}_test_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), test)

    if add_dayofweek:
        dayofweek_train = dayofweek[: train_steps]
        dayofweek_val = dayofweek[train_steps:train_steps+val_steps]
        dayofweek_test = dayofweek[-test_steps:]

        np.save("{}{}_train_dow_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), dayofweek_train)
        np.save("{}{}_val_dow_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), dayofweek_val)
        np.save("{}{}_test_dow_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), dayofweek_test)

    if add_timeofday:
        timeofday_train = timeofday[: train_steps]
        timeofday_val = timeofday[train_steps:train_steps+val_steps]
        timeofday_test = timeofday[-test_steps:]

        np.save("{}{}_train_tod_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), timeofday_train)
        np.save("{}{}_val_tod_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), timeofday_val)
        np.save("{}{}_test_tod_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), timeofday_test)

def generate_train_val_test_csv(dataset_dir, dataset_name, train_ratio, test_ratio):
    df = pd.read_csv(os.path.join(dataset_dir, dataset_name)+".csv")
    traffic = df.values

    num_step = df.shape[0]
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = traffic[:train_steps]
    val = traffic[train_steps:train_steps+val_steps]
    test = traffic[-test_steps:]

    np.save("{}{}_train_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), train)
    np.save("{}{}_val_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), val)
    np.save("{}{}_test_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), test)

def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)

def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)

def get_matrix(file_path, Ks):
    W = pd.read_csv(file_path, header=None).values.astype(float)
    L = scaled_laplacian(W)
    Lk = cheb_poly(L, Ks)
    Lk = torch.Tensor(Lk.astype(np.float32))
    return Lk

def calc_gso(dir_adj, gso_type):
    dir_adj = sp.load_npz(dir_adj)
    dir_adj = dir_adj.tocsc()
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')
    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    eigval_max = max(eigsh(A=gso, k=6, which='LM', return_eigenvectors=False))

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id
    return gso
