#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt

model_params = {'tau': 0.025,
                'nstep': 1000,
                'N': 3,
                'd': 3}

res_params = {'radius':0.4,
             'degree': 100,
             'sigma': 1.0,
             'Dr': 1000,
             'train_length': 10000,
             'predict_length': 5000,
            'num_inputs': model_params['N'],
             'beta': 0.00001
              }
# Randomly generated reservoir networks
def generate_reservoir(size,radius,degree):
    sparsity = degree/float(size);
    A = sparse.rand(size,size,density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A/e) * radius
    return A

# The initial reservoir state space is activated using a nonlinear activation function. The procedure reads the training data to activate.
def reservoir_layer(A, Win, input, res_params):
    # Assigning values to the initial state space of the reservoir
    states = np.zeros((res_params['train_length'],res_params['Dr']))
    states[0] = np.zeros(res_params['Dr'])
    # Cyclic update of reservoir state space by training time
    for i in range(1, res_params['train_length']):
        states[i] = np.tanh(A.dot(states[i-1]) + Win.dot(input[i-1]) )
    # Retain the last moment state vector of the training time for subsequent predictions
    states_nearfuture = np.tanh( A.dot(states[res_params['train_length']-1]) + Win.dot(input[res_params['train_length']-1]) )
    return states,states_nearfuture

# Training reservoir phase
def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['Dr'], res_params['radius'], res_params['degree'])
    q = int(res_params['Dr']/res_params['num_inputs'])
    Win = np.zeros((res_params['Dr'],res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i*q: (i+1)*q,i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1,q)[0])
    states,states_nearfuture = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    return states_nearfuture, Wout, A, Win

# Solve the weight matrix
def train(res_params,states,data):
    beta = res_params['beta']
    RC_states = np.hstack( (states, np.power(states, 2), np.power(states, 3) ,np.power(states, 4),np.power(states, 5)) )
    Wout =(data[0:res_params['train_length']].T).dot(RC_states).dot (np.linalg.inv(RC_states.T.dot(RC_states) + beta * np.eye(5 * res_params['Dr'])))
    return Wout

# Forecasting phase
def predict(A, Win, res_params, states_nearfuture, Wout):
    output = np.zeros((res_params['predict_length'], res_params['Dr']))
    output_states = np.zeros((res_params['predict_length'], 5*res_params['Dr']))
    output[0] = states_nearfuture
    output_states[0] = np.hstack( (output[0], np.power(output[0], 2),np.power(output[0], 3),np.power(output[0], 4),np.power(output[0], 5)) )
    # Autonomous forecasting phase
    for i in range(1,res_params['predict_length']):
        output[i] = np.tanh(A.dot(output[i-1]) + Win.dot( Wout.dot(output_states[i-1].T) ) )
        output_states[i] = np.hstack( (output[i], np.power(output[i], 2),np.power(output[i], 3),np.power(output[i], 4),np.power(output[i], 5)) )
    predict = Wout.dot(output_states.T)
    return predict

data = np.load('./data/lorenz_63.npy')

states_nearfuture,Wout,A,Win = train_reservoir(res_params,data[:res_params['train_length'],:])

output = predict(A, Win,res_params,states_nearfuture,Wout)
