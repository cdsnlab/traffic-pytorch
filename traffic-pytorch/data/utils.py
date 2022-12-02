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
    num_nodes = df.shape[1]
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

# Get the nmatrix from the data
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

def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


# TOSDO: combine with other csv dataset
def generate_train_val_test_with_matrix(dataset_dir, dataset_name, train_ratio, test_ratio, sigma1=0.1, sigma2=10, thres1=0.6, thres2=0.5):
    """
    read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    :param sigma1: float, default=0.1, sigma for the semantic matrix
    :param sigma2: float, default=10, sigma for the spatial matrix
    :param thres1: float, default=0.6, the threshold for the semantic matrix
    :param thres2: float, default=0.5, the threshold for the spatial matrix

    :output data: tensor, T * N * 1
    :output dtw_matrix: array, semantic adjacency matrix
    :output sp_matrix: array, spatial adjacency matrix
    """
    df = pd.read_csv(os.path.join(dataset_dir, dataset_name)+".csv")
    num_node = df.shape[1]
    traffic = df.values
    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)
    num_step = df.shape[0]
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = traffic[:train_steps]
    val = traffic[train_steps:train_steps+val_steps]
    test = traffic[-test_steps:]

    np.save("{}/{}_train_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), train)
    np.save("{}/{}_val_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), val)
    np.save("{}/{}_test_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), test)

    data = traffic
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    # generate semantic adjacency matrix
    data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
    data_mean = data_mean.squeeze().T 
    dtw_distance = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(i, num_node):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
    for i in range(num_node):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]
    np.save(f'{dataset_dir}/{dataset_name}_dtw_distance.npy', dtw_distance)
    
    # generate spatial adjacency matrix
    with open(f'{dataset_dir}/{dataset_name}_distance.csv', 'r') as fp:
        dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
        file = csv.reader(fp)
        for line in file:
            break
        for line in file:
            start = int(line[0])
            end = int(line[1])
            dist_matrix[start][end] = float(line[2])
            dist_matrix[end][start] = float(line[2])
        np.save(f'data/{filename}_spatial_distance.npy', dist_matrix)

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0)/2/num_node}')
    print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0)/2/num_node}')
    return torch.from_numpy(data.astype(np.float32)), mean_value, std_value, dtw_matrix, sp_matrix