# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:44:58 2023
@author: Siyuan Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

## This is the main function for the TNN model
import numpy as np
import scipy.io as scio
import scipy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from util import TNSR
import scipy.stats as stats
import argparse

np.random.seed(1)
tf.set_random_seed(1)

parser = argparse.ArgumentParser(description='arguments for the TNN model')

# input arguments
parser.add_argument('--p', type=float, default=0.2, help='Sampling rate')
parser.add_argument('--sigma', type=float, default=0.1, help='Noise standard deviation')
parser.add_argument('--lamb', type=float, default=3, help='Regularization coefficient')
parser.add_argument('--runs', type=int, default=15000, help='Number of iterations')
parser.add_argument('--verbose', type=bool, default=False, help='Display the iteration')

args = parser.parse_args()

p = args.p
sigma = args.sigma
lamb = args.lamb
runs = args.runs
verbose = args.verbose

## load ssf data
data = scipy.io.loadmat('data.mat')
ssf = np.array(data['data']).astype('float32')
[N_x,N_y,N_z] = ssf.shape
# load the mean
mean = scipy.io.loadmat('data_mean.mat')
data_mean = np.array(mean['data_mean']).astype('float32')

## add noise
noisy_data = ssf + sigma * np.random.randn(N_x,N_y,N_z)

## normalize
ssp_max = np.max(np.abs(noisy_data))
x = noisy_data / ssp_max

## observation
nmod = 3
total = N_x * N_y * N_z
size = [N_x,N_y,N_z]
N = (int)(p*np.prod(size))
index = []
for i in range(N_x):
    for j in range(N_y):
        for k in range(N_z):
            index.append([i,j,k])
index = np.array(index)
ind = index[np.random.choice(total,N,replace=False)]#

ob = np.zeros(size)
for i in range(N):
    ob[tuple(ind[i,:])] = 1

x_ob = x * ob


## TNN model
# Input
I=N_x;J=N_y;K=N_z
input_shape = [I,J,K]
R1=5;R2=5;R3=5
rank = [R1,R2,R3]
z = np.random.randn(R1,R2,R3)
z = tf.Variable(z, dtype=tf.float32)

# decoder
v2d = [stats.truncnorm(-1, 1).rvs([10,5]) for i in range(3)]
V2d = [tf.Variable(v2d[k], dtype=tf.float32) for k in range(3)]
zd = tf.nn.relu(TNSR.tucker_to_tensor(z,V2d))

vd = [stats.truncnorm(-1, 1).rvs([input_shape[i],10]) for i in range(3)]
Vd = [tf.Variable(vd[k], dtype=tf.float32) for k in range(3)]
x_hat = tf.nn.tanh(TNSR.tucker_to_tensor(zd,Vd))

## TV regularizer
def total_variation(images):
    sum_axis = None
    pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
    pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
    pixel_dif3 = images[:, :, 1:] - images[:, :, :-1]
    tot_var = (
        tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) +
        tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis) +
        tf.reduce_sum(tf.abs(pixel_dif3), axis=sum_axis) )
    return tot_var

## training settings
# Defining the loss function
reconstr_loss = tf.losses.mean_squared_error(x_ob,x_hat*ob)
cost = reconstr_loss + lamb * 1e-7 * total_variation(x_hat)
# Use ADAM optimizer
optimizer =  tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.9, beta2=0.999).minimize(cost)

## training function
x_re = 0
def train():
    # Training the network
    print('start training')
    global x_re 
    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    loss_train = []
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(runs):
            _,d,x_re = sess.run((optimizer, cost,x_hat))
                
            # Display logs per step
            if epoch % 100 == 0 and verbose:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(d))
                loss_train.append(d)

    return loss_train

## save the result
def save():
    # save
    dataNew = 'ssp_tnn.mat'
    ssp1 = x_re * ssp_max + data_mean
    scio.savemat(dataNew, {'ssp_tnn':ssp1})


if __name__ == '__main__':
    train()
    rmse = np.sqrt(np.mean(np.square(x_re * ssp_max - ssf)))
    print('RMSE: ', "{:.2f}".format(rmse))
    save()
