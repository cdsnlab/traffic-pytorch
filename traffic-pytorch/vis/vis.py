import matplotlib.pyplot as plt 
import pickle 
from tqdm import tqdm 

with open('../../datasets/bj-flow.pickle', "rb") as file:
    data = pickle.load(file)

data = data['train']
data = data.reshape(data.shape[0], 32, 32, 2)
data = data[1000:1050]

for i in tqdm(range(data.shape[0])):
    plt.figure(figsize=(8,4))

    plt.subplot(1, 2, 1) 
    plt.imshow(data[i, :, :, 0])
    plt.title('Inflow')

    plt.subplot(1, 2, 2) 
    plt.imshow(data[i, :, :, 1])
    plt.title('Outflow')

    plt.tight_layout()
    plt.savefig('imgs/{}.png'.format(i))
    plt.clf()