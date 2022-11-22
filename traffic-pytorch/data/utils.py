import pickle 
import numpy as np
import os
import torch
import pandas as pd


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