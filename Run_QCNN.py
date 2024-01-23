# %%
import pennylane as qml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import autograd.numpy as anp

from tqdm import tqdm
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.providers.aer import QasmSimulator

# %%
#IBMQ.save_account(YOUR TOKEN)
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-skku', group='yonsei-uni', project='yu-students')

# %%
device = qml.device('qiskit.aer', wires = 2)
@qml.qnode(device)
def Circuit(param, cfg, fixed = None):
    qml.AmplitudeEmbedding(fixed, wires=[0,1], normalize = True)
    param_idx = 0
    wire1 = cfg['wire1']
    wire2 = cfg['wire2']
    for idx, (element1, element2) in enumerate(zip(wire1, wire2)):
        if element1 == 'C' or element2 == 'C':
            query = element1+element2 
            if query == 'CC':
                qml.SWAP([0,1])
            elif query == 'CX':
                qml.CRX(param[param_idx], [0,1])
                param_idx += 1
            elif query == 'XC':
                qml.CRX(param[param_idx], [1,0])
                param_idx += 1
            elif query == 'CY':
                qml.CRY(param[param_idx], [0,1])
                param_idx += 1
            elif query == 'YC':
                qml.CRY(param[param_idx], [1,0])
                param_idx += 1
            elif query == 'CZ':
                qml.CRZ(param[param_idx], [0,1])
                param_idx += 1
            elif query == 'ZC':
                qml.CRZ(param[param_idx], [1,0])
                param_idx += 1
        else:
            if element1 == 'X':
                qml.RX(param[param_idx], 0)
                param_idx += 1
            if element1 == 'Y':
                qml.RY(param[param_idx], 0)
                param_idx += 1
            if element1 == 'Z':
                qml.RX(param[param_idx], 0)
                param_idx += 1
            if element1 == 'H':
                qml.Hadamard(0)
            if element2 == 'X':
                qml.RX(param[param_idx], 1)
                param_idx += 1
            if element2 == 'Y':
                qml.RY(param[param_idx], 1)
                param_idx += 1
            if element2 == 'Z':
                qml.RX(param[param_idx], 1)
                param_idx += 1
            if element2 == 'H':
                qml.Hadamard(1)
            
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

# %%
torch.manual_seed(42)
np.random.seed(42)

train_df = pd.read_csv('PCA_mnist_train.csv')
valid_df = pd.read_csv('PCA_mnist_valid.csv')
test_df = pd.read_csv('PCA_mnist_test.csv')


train_df = train_df[(train_df['label'] == float(0)) | (train_df['label'] == float(1))]
valid_df = valid_df[(valid_df['label'] == float(0)) | (valid_df['label'] == float(1))]
test_df = test_df[(test_df['label'] == float(0)) | (test_df['label'] == float(1))]

train_X = train_df.drop('label', axis = 1)
train_y = train_df['label'].to_numpy()
valid_X = valid_df.drop('label', axis = 1)
valid_y = valid_df['label'].to_numpy()
test_X = test_df.drop('label', axis = 1)
test_y = test_df['label'].to_numpy()

train_y_ = torch.tensor(train_y, dtype = int)
valid_y_ = torch.tensor(valid_y, dtype = int)
test_y_ =torch.tensor(test_y, dtype = int)
train_y_hot = F.one_hot(train_y_)
valid_y_hot = F.one_hot(valid_y_)
test_y_hot =F.one_hot(test_y_)
train_X_ = train_X.to_numpy()
valid_X_ = valid_X.to_numpy()
test_X_ = test_X.to_numpy()

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
        
    loss = loss / len(labels)
    return loss

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss

def softmax(x):
    
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def cost(params, X, Y, cfg, cost_fn):
    predictions = [Circuit(params, cfg, x).argmax() for x in X]
    if cost_fn == 'mse':
        loss = square_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    return loss

# Circuit training parameters
hpm = {
    'steps': 50,
    'train_batch_size': 20,
    'valid_batch_size' : 500,
    'test_batch_size' : 2000
}

def count_params(cfg):
    wire1 = cfg['wire1']
    wire2 = cfg['wire2']
    n_params = 0
    for (element1, element2) in zip(wire1, wire2):
        if element1 in ['X','Y','Z']:
            n_params += 1
        if element2 in ['X','Y','Z']:
            n_params += 1
    return n_params

def circuit_training(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, cost_fn='mse'):
    wire1 = ['H', 'C', 'X', 'C', 'Z']
    wire2 = ['H', 'X', 'C', 'Z', 'C']
    architecture = ''
    for element in wire1+wire2:
        architecture += element
        
    with open(f"/home/hahajjjun/Junha Park/QNAS/log_{architecture}.txt", "w") as file:
    
        cfg = {'wire1':wire1, 'wire2':wire2}
        n_params = count_params(cfg)
        params =  qml.numpy.random.randn(n_params)
        params.requires_grad = True
        
        best = np.inf
        best_param = params
        for it in tqdm(range(hpm['steps'])):

            train_batch_index = np.random.randint(0, len(X_train), (hpm['train_batch_size'],))
            valid_batch_index = np.random.randint(0, len(X_valid), (hpm['valid_batch_size'],))
            test_batch_index = np.random.randint(0, len(X_test), (hpm['test_batch_size'],))

            X_batch_train = [X_train[i] for i in train_batch_index]
            Y_batch_train = [Y_train[i] for i in train_batch_index]
            X_batch_valid = [X_valid[i] for i in valid_batch_index]
            Y_batch_valid = [Y_valid[i] for i in valid_batch_index]
            X_batch_test = [X_test[i] for i in test_batch_index]
            Y_batch_test = [Y_test[i] for i in test_batch_index]

            '''
            Parameter Update via COBYLA
            '''
            out = minimize(lambda v : cost(params, X_batch_train, Y_batch_train, cfg, cost_fn), x0=params, method="COBYLA")
            params = out['x']

            val_loss = cost(params, X_batch_valid, Y_batch_valid, cfg, cost_fn)
            if val_loss < best:
                best_param = params
                best = val_loss
            file.write(f'iteration : {it}, valid loss : {val_loss}, param : {params} \n')


        test_loss = cost(best_param, X_batch_test, Y_batch_test, cfg, cost_fn)
        file.write(f'valid best score : {best}, valid best parameter : {best_param}, test score : {test_loss} \n')
        
    return

circuit_training(train_X_, train_y, valid_X_, valid_y, test_X_, test_y)