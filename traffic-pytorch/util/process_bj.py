import h5py
import numpy as np
import pickle 

f = h5py.File('bj/BJ_FLOW.h5', 'r')
data = np.array(f['data'])
f.close()

days, hours, rows, cols, _ = data.shape
data = np.reshape(data, (days * hours, rows * cols, -1))

n_timestamp = data.shape[0]
num_train = int(n_timestamp * 0.7)
num_val = int(n_timestamp * 0.2)
num_test = n_timestamp - num_train - num_val

train, val, test = data[:num_train], data[num_train: num_train + num_val], data[-num_test:]

f = h5py.File('bj/BJ_FEATURE.h5', 'r')
node = np.array(f['embeddings'])
f.close()

rows, cols, num_feats = node.shape
node = np.reshape(node, (rows*cols, num_feats))

f = h5py.File('bj/BJ_GRAPH.h5', 'r')
edge = np.array(f['data'])
f.close()

bj_data = {
    "train": train,
    "val": val,
    "test": test, 
    "node": node, 
    "edge": edge
}

with open("bj-flow.pickle","wb") as fw:
    pickle.dump(bj_data, fw)