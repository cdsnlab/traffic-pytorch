import pickle 
import numpy as np
import os
import torch
import pandas as pd
import scipy.sparse as sp
from fastdtw import fastdtw
import csv

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
        x[i] = data[i: i + num_his, :]
        y[i] = data[i + num_his: i + num_his + num_pred, :]
    return x, y

# TODO: finishe preprocessing for multiple velocity comp
def seq2instance_3d(data, num_his, num_pred):
    num_step, dims, vel_feat = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros((num_sample, num_his, dims, vel_feat))
    y = np.zeros((num_sample, num_pred, dims, vel_feat))
    for i in range(num_sample):
        x[i] = data[i: i + num_his, :, :]
        y[i] = data[i + num_his: i + num_his + num_pred, :, :]
    return x, y

def generate_train_val_test(dataset_dir, dataset_name, train_ratio, test_ratio, add_dayofweek=False, add_timeofday=False, format='h5'):
    if format == 'h5':
        df = pd.read_hdf(os.path.join(dataset_dir, dataset_name)+".h5")
    elif format == 'csv':
        df = pd.read_csv(os.path.join(dataset_dir, dataset_name)+".csv")
    elif format == 'npz':
        df = np.load(os.path.join(dataset_dir, dataset_name)+".npz")['data']

    num_nodes = df.shape[1]
    if format == 'npz':
        traffic = df
    else:
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

    if add_dayofweek:
        time = pd.DatetimeIndex(df.index)
        dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1)).numpy()
        dayofweek_train = dayofweek[: train_steps]
        dayofweek_val = dayofweek[train_steps:train_steps+val_steps]
        dayofweek_test = dayofweek[-test_steps:]

        np.save("{}{}_train_dow_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), dayofweek_train)
        np.save("{}{}_val_dow_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), dayofweek_val)
        np.save("{}{}_test_dow_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), dayofweek_test)

    if add_timeofday:
        timeofday = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1)).numpy()
        timeofday_train = timeofday[: train_steps]
        timeofday_val = timeofday[train_steps:train_steps+val_steps]
        timeofday_test = timeofday[-test_steps:]

        np.save("{}{}_train_tod_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), timeofday_train)
        np.save("{}{}_val_tod_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), timeofday_val)
        np.save("{}{}_test_tod_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), timeofday_test)


def generate_data_matrix(dataset_dir, dataset_name, train_ratio, test_ratio, sigma1=0.1, sigma2=10, thres1=0.6, thres2=0.5):
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
    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)    
    data = np.load(os.path.join(dataset_dir, dataset_name)+".npz")['data']
    # use a small part of the data
    #data = data[:24*12*3, :, :]
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    num_step = data.shape[0]
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = data[:train_steps]
    val = data[train_steps:train_steps+val_steps]
    test = data[-test_steps:]

    np.save("{}{}_train_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), train)
    np.save("{}{}_val_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), val)
    np.save("{}{}_test_{}_{}.npy".format(dataset_dir, dataset_name, train_ratio, test_ratio), test)

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
    np.save("{}/{}_dtw_distance.npy".format(dataset_dir, dataset_name), dtw_distance)
    
    # generate spatial adjacency matrix
    with open("{}/{}.csv".format(dataset_dir, dataset_name), 'r') as fp:
        dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
        file = csv.reader(fp)
        for line in file:
            break
        for line in file:
            start = int(line[0])
            end = int(line[1])
            dist_matrix[start][end] = float(line[2])
            dist_matrix[end][start] = float(line[2])
        np.save("{}/{}_spatial_distance.npy".format(dataset_dir, dataset_name), dist_matrix)


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
    print(W.shape)
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


def generate_adjacency_matrix(dataset_dir, dataset_name, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                    dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                        dtype=np.float32)
    distance_df_filename  = dataset_dir + '/' + dataset_name + '.csv'
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}

        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distaneA[id_dict[i], id_dict[j]] = distance
    else:
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distaneA[i, j] = distance
    np.save("{}/{}_adjacency.npy".format(dataset_dir, dataset_name), A)
    np.save("{}/{}_distance.npy".format(dataset_dir, dataset_name), distaneA)


