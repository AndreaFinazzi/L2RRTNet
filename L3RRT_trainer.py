# This file aims to train the "controller" U network
# The network, trained by "best" trajectories from already solved prolems, 
# aims to predict the next best action given a specific problem setting and the current latent state.
# This trainer solves a set of problems, stores the trajectories represented as a sequence of control input and then trains a model to predict the next state


from __future__ import absolute_import, division #, print_function
import tensorflow as tf
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os, pathlib
import csv
from random import randint, random
import time
import math
from heapq import heappush, heappop

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
file_path = pathlib.Path(__file__).parent.absolute()
file_path_str = str(file_path)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True

# setup dimensions and training specs-
# x, u, z, data, mb, lr
img_res = 32
img_dim = img_res*img_res

x_dim = img_dim
y_dim = img_dim
u_dim = 2 # control effort
z_dim = 2
data_dim = 3*img_dim + u_dim # x_i, x_i+1, x_empty (to pass through obstacles), control input (x,y unit vector)

mb_size = 128
lr = 1e-4

h_CC_dim = 128
conv_CC_filters = 10
conv_CC_filter_width = 6
pool_CC_stride = 1

tf.compat.v1.reset_default_graph() 

#### COPY FROM HERE
z_t_U = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim], name="z_t_U")
z_goal_U = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim], name="z_goal_U")
# Flat representation of xempty (generated from the higher lever CNN)
# dense_CC_in_U = tf.compat.v1.placeholder(tf.float32, shape=[None, 4*h_CC_dim], name="dense_CC_in_U")
dense_CC_conv = tf.compat.v1.placeholder(tf.float32, shape=[None, 4*h_CC_dim], name="dense_CC_conv")
u_U = tf.compat.v1.placeholder(tf.float32, shape=[None, u_dim], name="u_U")

# CC hoping in transfer learning
zs_CC = tf.concat(axis=1, values=[z_t_U, z_goal_U])
dense_CC_zs = tf.layers.dense(inputs=zs_CC, units=4*h_CC_dim, activation=tf.nn.relu, name="dense_CC_zs")

inputs_CC = tf.concat(axis=1, values=[dense_CC_zs, dense_CC_conv]) 
dense_CC1 = tf.layers.dense(inputs=inputs_CC, units=h_CC_dim, activation=tf.nn.relu, name="dense_CC1")
dropout_CC1 = tf.layers.dropout(inputs=dense_CC1, rate=0.5)
dense_CC2 = tf.layers.dense(inputs=dropout_CC1, units=h_CC_dim, activation=tf.nn.relu, name="dense_CC2")
dropout_CC2 = tf.layers.dropout(inputs=dense_CC2, rate=0.5)
dense_CC3 = tf.layers.dense(inputs=dropout_CC2, units=h_CC_dim, activation=tf.nn.relu, name="dense_CC3")
dropout_CC3 = tf.layers.dropout(inputs=dense_CC3, rate=0.5)
dense_CC4 = tf.layers.dense(inputs=dropout_CC3, units=h_CC_dim, activation=tf.nn.relu, name="dense_CC4")
dropout_CC4 = tf.layers.dropout(inputs=dense_CC4, rate=0.5)
dense_CC5 = tf.layers.dense(inputs=dropout_CC4, units=h_CC_dim, activation=tf.nn.relu, name="dense_CC5")


# U Network definition

dense_U1 = tf.layers.dense(inputs=dense_CC5, units=h_CC_dim, activation=tf.nn.relu, name="dense_U1")
dropout_U1 = tf.layers.dropout(inputs=dense_U1, rate=0.5)
dense_U2 = tf.layers.dense(inputs=dropout_U1, units=h_CC_dim/2, activation=tf.nn.relu, name="dense_U2")
dropout_U2 = tf.layers.dropout(inputs=dense_U2, rate=0.5)
dense_U3 = tf.layers.dense(inputs=dropout_U2, units=h_CC_dim/4, activation=tf.nn.relu, name="dense_U3")
dropout_U3 = tf.layers.dropout(inputs=dense_U3, rate=0.5)

zs_U = tf.concat(axis=1, values=[z_t_U, z_goal_U])
dense_U_zs = tf.layers.dense(inputs=zs_U, units=h_CC_dim/4, activation=tf.nn.relu, name="dense_U_zs")

inputs_U = tf.concat(axis=1, values=[dense_U_zs, dropout_U3]) 
# May need an intermediate layer
dense_U4 = tf.layers.dense(inputs=inputs_U, units=h_CC_dim/8, activation=tf.nn.relu, name="dense_U4")
dropout_U4 = tf.layers.dropout(inputs=dense_U4, rate=0.5)
dense_U5 = tf.layers.dense(inputs=dropout_U4, units=h_CC_dim/16, activation=tf.nn.relu, name="dense_U5")

y_U = tf.layers.dense(inputs=dense_U5, units=u_dim, name="y_U")

control_loss = tf.losses.mean_squared_error(labels=u_U, predictions=y_U)
U_trainable_variables = [v for v in tf.trainable_variables() if 'CC' not in v.name.split(':')[0]]

train_step_U = tf.train.AdamOptimizer(lr).minimize(control_loss, 
                var_list=U_trainable_variables)

mean_loss = tf.reduce_mean(control_loss)
#### COPY DOWN TO HERE


## Initialize
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
it_U = 0

print('initialization: done')

# Restore what's restorable
reader = tf.train.NewCheckpointReader(file_path_str + "/model/LSBMP_geometric.ckpt")
restore_dict = dict()
for v in tf.trainable_variables():
    tensor_name = v.name.split(':')[0]
    if reader.has_tensor(tensor_name):
        restore_dict[tensor_name] = v
        # print('has tensor ', tensor_name)

saver = tf.train.Saver(restore_dict)
saver.restore(sess, file_path_str + "/model/LSBMP_geometric.ckpt")
print("Model restored (as best I could).")


### Time to train the model

# 1. Load training and testing datasets
dataset_row_dim = 2*z_dim + 4*h_CC_dim + u_dim
data_train = []
data_test = []


filename = file_path_str + '/data/controller_data_train_5000.csv' 
with open(filename, 'rt') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        row_data = map(float,row[0:dataset_row_dim])
        data_train.append(list(row_data))

filename = file_path_str + '/data/controller_data_test_5000.csv' 
with open(filename, 'rt') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        row_data = map(float,row[0:dataset_row_dim])
        data_test.append(list(row_data))

### WARNING: VALID UNTIL A PROPER DATASET IS GENERATED!!!
# data_train.extend(data_test)
# dataset_train = np.array(data_train, dtype='d')[:-100]
# dataset_test = dataset_train[-100:]

### WARNING: RESTORE THESE TWO LINES AS SOON AS THE FINAL DATASET IS LOADED
dataset_train = np.array(data_train, dtype='d')
dataset_test = np.array(data_test, dtype='d')

print('Read in ', dataset_train.shape[0], ' training rows')
print('Read in ', dataset_test.shape[0], ' testing rows')

# Shuffle the datasets
shuffler = [randint(0,dataset_train.shape[0]-1) for n in range(0,dataset_train.shape[0])]
dataset_train_r = dataset_train[shuffler,:]
shuffler = [randint(0,dataset_test.shape[0]-1) for n in range(0,dataset_test.shape[0])]
dataset_test_r = dataset_test[shuffler,:]


# 2. Train the model
EPOCHS = 1000
TRAIN_BATCH_SIZE = 15
ITERATIONS = int(np.floor(dataset_train.shape[0]/TRAIN_BATCH_SIZE))

current_batch = np.zeros((TRAIN_BATCH_SIZE, dataset_row_dim))

for epoch in range(0,EPOCHS):

    for iteration in range(0,ITERATIONS):
        current_batch = dataset_train_r[iteration*TRAIN_BATCH_SIZE:(iteration+1)*TRAIN_BATCH_SIZE,:]

        _, train_loss = sess.run([train_step_U, control_loss],
            feed_dict={
                z_t_U : current_batch[:,:z_dim], 
                z_goal_U : current_batch[:,z_dim:2*z_dim], 
                dense_CC_conv : current_batch[:,2*z_dim:-u_dim], 
                u_U : current_batch[:,-u_dim:] })


    # Evaluate:
    test_loss, test_prediction, test_loss_mean = sess.run([control_loss, y_U, mean_loss],
            feed_dict={
                z_t_U : dataset_test_r[:,:z_dim], 
                z_goal_U : dataset_test_r[:,z_dim:2*z_dim], 
                dense_CC_conv : dataset_test_r[:,2*z_dim:-u_dim], 
                u_U : dataset_test_r[:,-u_dim:] })

    print("Epoch ", epoch, ": Mean Loss of test batch: ", test_loss_mean)

# Create a saver.
saver = tf.compat.v1.train.Saver(U_trainable_variables)
saver.save(sess, file_path_str + '/model/U_network-v2')