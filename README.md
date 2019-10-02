Knockoff via Variational Auto-encoder
=====================================

This repository provide python and R code for training via deep latent variable model. 
Accompanying paper: https://arxiv.org/abs/1809.10765

### Software dependencies.
The knockoff generator is writen in python with dependency in pytorch. 
The code 
- python=3.6.5
- pytorch=
- numpy=

The simulation in linear and logistic lasso regression for FDR control is written in R.

## Tutorial
### To use the main class for VAE knockoff generator in vae_plain, the VAE Class incoorperate flexible user defined layers for decoder and endocer. to initiate provide:

##### X_dim
Dimension for X
##### zdim
dimension for Z: 
##### nhin
vector for numbers of neurals in hidden layers for encoder. e.g. [ ] for no hidden layer, [100] for one hidden layer with 100 neuron, [100, 200] for two hidden layers with 100 and 200 neuron respectively.
##### nout 
vector for numbers of neurals in hidden layers for decoder
An Example Code using the Class:

```
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
from vae_plain import VAE
from scipy.linalg import cholesky
bin=0
n_epoch = 30
p = 100
m = 200
mb_size = 25  # training batch size

m=100
p=50

X = norm.rvs(size=(m, p))
X_tr = (X.dot(C) > 0) * 1.
X_tr=torch.from_numpy(X_tr)  
model= VAE(X_dim,zdim,nhin,nhout)
optimizer=optim.Adam(model.parameters())

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
    
    
for epoch in range(1,n_epoch):
    train(epoch,X_tr,col2=0.2)
    
    
```
#### To try some simulation settings in our paper
cd path/to/transimu.py 

python trainsimu.py 6 Xka zdim=100 nhin=[100] nhout=[100] m=100 p=50 n_epoch=50

### License
