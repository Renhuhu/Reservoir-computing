#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
# import pandas as pd
import time
import matplotlib.pyplot as plt

shift_k = 0

approx_res_size = 1000


model_params = {'tau': 0.025,
                'nstep': 1000,
                'N': 3,
                'd': 3}

res_params = {'radius':0.4,
             'degree': 100,
             'sigma': 1.0,
             'train_length': 10000,
             'N': int(np.floor(approx_res_size/model_params['N']) * model_params['N']),
             'num_inputs': model_params['N'],
             'predict_length': 5000,
             'beta': 0.00001
              }

def generate_reservoir(size,radius,degree):
    sparsity = degree/float(size);
    A = sparse.rand(size,size,density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A/e) * radius
    return A

def reservoir_layer(A, Win, input, res_params):
    states = np.zeros((res_params['N'],res_params['train_length']))
    for i in range(res_params['train_length']-1):
        states[:,i+1] = np.tanh(np.dot(A,states[:,i]) + np.dot(Win,input[:,i]))
    return states

def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['N'], res_params['radius'], res_params['degree'])
    q = int(res_params['N']/res_params['num_inputs'])
    Win = np.zeros((res_params['N'],res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i*q: (i+1)*q,i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1,q)[0])
    states = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    x = states[:,-1]
    return x, Wout, A, Win

def train(res_params,states,data):
    beta = res_params['beta']
    idenmat = beta * sparse.identity(res_params['N'])
    states2 = states.copy()
    for j in range(2,np.shape(states2)[0]-2):
        if (np.mod(j,2)==0):
            states2[j,:] = (states[j-1,:]*states[j-2,:]).copy()
    U = np.dot(states2,states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv,np.dot(states2,data.transpose()))
    return Wout.transpose()

def predict(A, Win, res_params, x, Wout):
    output = np.zeros((res_params['num_inputs'],res_params['predict_length']))
    for i in range(res_params['predict_length']):
        x_aug = x.copy()
        for j in range(2,np.shape(x_aug)[0]-2):
            if (np.mod(j,2)==0):
                x_aug[j] = (x[j-1]*x[j-2]).copy()
        out = np.squeeze(np.asarray(np.dot(Wout,x_aug)))
        output[:,i] = out
        x1 = np.tanh(np.dot(A,x) + np.dot(Win,out))
        x = np.squeeze(np.asarray(x1))
    return output, x

data = np.load('lorenz_63.npy')
data = np.transpose(data)

x,Wout,A,Win = train_reservoir(res_params,data[:,shift_k:shift_k+res_params['train_length']])

output, _ = predict(A, Win,res_params,x,Wout)
output_data = np.transpose(output)

