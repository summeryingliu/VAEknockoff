import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
from vae_plain import VAE
from scipy.linalg import cholesky

import sys

#must input setting, knockoff name, zdim, nhin, nhout
setting=eval(sys.argv[1])
print(setting)
namek=sys.argv[2]

bin=0
n_epoch = 30
if setting==3:
    bin=1
p = 100
m = 200
mb_size = 25  # training batch size


for arg in sys.argv[3:]:
    exec(arg)

X_dim = p
'''zdim=50
nhin=[]
nhout=[]'''
folder='C:/Users/Ying Liu/PycharmProjects/latentoutput/'

np.random.seed(137)
n = m

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model= VAE(X_dim,zdim,nhin,nhout)#.to(device) #move the model parameter to the GPU
optimizer=optim.Adam(model.parameters())

torch.save(model, folder+'setting'+str(setting)+'/'+namek+'.pth')

#SIGMA = 0.1*np.ones([2*p,2*p])
if setting==2:
    SIGMA = 0.1*np.ones([2*p,2*p])
    np.fill_diagonal(SIGMA, 1)
    C = cholesky(SIGMA, lower=True)
def generatortraining2(seed):
    np.random.seed(i)
    Z = np.random.uniform(size=(m, 2 * p)).dot(C)
    X_tr = np.zeros([m, p])
    X_tr[:, range(0, p, 2)] = Z[:, range(0, 2 * p, 4)] + 0.5 * np.power(Z[:, range(1, 2 * p, 4)], 3)
    X_tr[:, range(1, p, 2)] = Z[:, range(2, 2 * p, 4)] + 0.5 * np.exp(Z[:, range(3, 2 * p, 4)]) - 0.5 * np.power(
        Z[:, range(2, 2 * p, 4)], 2)
    X_tr = (X_tr - np.min(X_tr, axis=0)) / (np.max(X_tr, axis=0) - np.min(X_tr, axis=0))
    X_tr = torch.from_numpy(X_tr)
    return X_tr
    # generating training data
if setting==1:
    np.random.seed(137)
    z_dim=200
    h1=150
    I1 = np.random.random_integers(0, z_dim - 1, h1)
    I2 = np.random.random_integers(0, z_dim - 1, h1)
    I3 = np.random.random_integers(0, z_dim - 1, h1)
    W1 = np.random.random_integers(-2,2 , size=[z_dim, h1])
    W2 = np.random.random_integers(-2, 2, size=[h1, X_dim])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generatortraining1(seed):
    np.random.seed(i)
    Z_tr = np.random.normal(size=(n, z_dim))
    #H1 = Z_tr.dot(W1) + Z_tr[:, I1] * Z_tr[:, I1] + 2 * np.log(np.abs(Z_tr[:, I1]) + 1) + Z_tr[:, I1] / np.maximum(
    #    Z_tr[:, I2], 0.5)+np.random.normal(size=())
    H1= np.tanh(Z_tr.dot(W1))
    X_tr = sigmoid(H1.dot(W2))+0.5*np.random.normal(size=(n,X_dim))
    X_tr = (X_tr - np.min(X_tr, axis=0)) / (np.max(X_tr, axis=0) - np.min(X_tr, axis=0))
    X_tr = torch.from_numpy(X_tr)
    return X_tr

if setting==3:
    np.random.seed(137)
    SIGMA = 0.1*np.ones([p,p])
    np.fill_diagonal(SIGMA, 1)
    C = cholesky(SIGMA, lower=True)
    from scipy.stats import norm

def generatortraining3(seed):
    np.random.seed(i)
    X = norm.rvs(size=(m, p))
    X_tr = (X.dot(C) > 0) * 1.
    X_tr=torch.from_numpy(X_tr)
    return X_tr

if setting==4:
    SIGMA = 0.5*np.ones([2*p,2*p])
    np.fill_diagonal(SIGMA, 1)
    C = cholesky(SIGMA, lower=True)

def generatortraining4(seed):
    np.random.seed(i)
    Z = np.random.normal(size=(m, 2 * p)).dot(C)
    X_tr = np.zeros([m, p])
    X_tr[:, range(0, p, 2)] = 1.0*np.logical_xor(Z[:, range(0, 2 * p, 4)]>0, Z[:, range(1, 2 * p, 4)]>0)
    X_tr[:, range(1, p, 2)] = (np.cos(Z[:, range(2, 2 * p, 4)]) + np.sin(Z[:, range(3, 2 * p, 4)] - 0.5 * np.power(
        Z[:, range(2, 2 * p, 4)], 2)))**2
    #X_tr[:, range(1, p, 2)] = (X_tr[:, range(1, p, 2)] - np.min(X_tr[:, range(1, p, 2)], axis=0)) / (np.max(X_tr[:, range(1, p, 2)], axis=0) - np.min(X_tr[:, range(1, p, 2)], axis=0))
    X_tr = (X_tr - np.min(X_tr, axis=0)) / (np.max(X_tr, axis=0) - np.min(X_tr, axis=0))
    X_tr = torch.from_numpy(X_tr)
    return X_tr
    # generating tr

    # generating training data
if setting==5:
    np.random.seed(137)
    z_dim=200
    h1=150
    I1 = np.random.random_integers(0, z_dim - 1, h1)
    I2 = np.random.random_integers(0, z_dim - 1, h1)
    I3 = np.random.random_integers(0, z_dim - 1, h1)
    W1 = np.random.random_integers(-2,2 , size=[z_dim, h1])
    W2 = np.random.random_integers(-2, 2, size=[h1, X_dim])

def generatortraining5(seed):
    np.random.seed(i)
    Z_tr = np.random.normal(size=(n, z_dim))
    #H1 = Z_tr.dot(W1) + Z_tr[:, I1] * Z_tr[:, I1] + 2 * np.log(np.abs(Z_tr[:, I1]) + 1) + Z_tr[:, I1] / np.maximum(
    #    Z_tr[:, I2], 0.5)+np.random.normal(size=())
    H1= np.tanh(Z_tr.dot(W1))
    X_tr = sigmoid(np.cos(H1.dot(W2)**2+0.5)**2-1)
    X_tr = (X_tr - np.min(X_tr, axis=0)) / (np.max(X_tr, axis=0) - np.min(X_tr, axis=0))
    X_tr = torch.from_numpy(X_tr)
    return X_tr

if setting==6:
    np.random.seed(137)
    z_dim=200
    h1=150
    I1 = np.random.random_integers(0, z_dim - 1, h1)
    I2 = np.random.random_integers(0, z_dim - 1, h1)
    I3 = np.random.random_integers(0, z_dim - 1, h1)
    W1 = np.random.random_integers(-2,2 , size=[z_dim, h1])
    W2 = np.random.random_integers(-2, 2, size=[h1, X_dim])

def generatortraining6(seed):
    np.random.seed(i)
    Z_tr = np.random.normal(size=(n, z_dim))
    #H1 = Z_tr.dot(W1) + Z_tr[:, I1] * Z_tr[:, I1] + 2 * np.log(np.abs(Z_tr[:, I1]) + 1) + Z_tr[:, I1] / np.maximum(
    #    Z_tr[:, I2], 0.5)+np.random.normal(size=())
    H1= np.tanh(Z_tr.dot(W1))
    X_tr = sigmoid(np.cos(10*H1.dot(W2)+0.5)**2-1)
    X_tr = (X_tr - np.min(X_tr, axis=0)) / (np.max(X_tr, axis=0) - np.min(X_tr, axis=0))
    X_tr = torch.from_numpy(X_tr)
    return X_tr

def train(epoch,X,col1=0,col2=0):
    model.train()
    train_loss=0
    for i in range((n - 1) // mb_size + 1):
        #         set_trace()
        start_i = i * mb_size
        end_i = start_i + mb_size
        xb = X[start_i:end_i]#.to(device)

        optimizer.zero_grad()
        loss = model.train_loss(xb,col1,col2)
        loss.backward()
        train_loss+=loss.item()
        optimizer.step()

    print('===> Epoch: {} Average loss: {:.4f}'.format(epoch,train_loss/n))



def reset(m):
    if isinstance(m,nn.Linear):
        nn.init.zeros_(m.bias)
        nn.init.normal_(m.weight,std=0.05)

for i in range(500):
    if setting==1:
        X_tr=generatortraining1(i)

    if setting==2:
        X_tr = generatortraining2(i)

    if setting==3:
        X_tr = generatortraining3(i)

    if setting==4:
        X_tr = generatortraining4(i)

    if setting == 5:
        X_tr = generatortraining5(i)

    if setting == 6:
        X_tr = generatortraining6(i)

    model.apply(reset)
    for epoch in range(1,n_epoch):
   # pdb.set_trace()
        train(epoch,X_tr,col2=0.2)

    if bin==0:
        xknock=model.generator(X_tr)

    if bin==1:
        xknock = model.generator(X_tr,bin=1)

    np.savetxt(folder+'setting'+str(setting)+'/X'+ str(i) + '.csv', X_tr.detach().numpy(), delimiter=',')
    np.savetxt(folder+'setting'+str(setting)+'/'+namek+ str(i) + '.csv', xknock.detach().numpy(), delimiter=',')