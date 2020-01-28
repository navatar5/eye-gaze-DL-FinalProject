# -*- coding: utf-8 -*-
"""
Pranesh Navarathna
ECSE 6965 Deep Learning
Final Project - Eye Gaze Position Estimation

Implemented in Tensorflow v1.0.1

This model takes in pictues of left eyes and predicts what position on a phone screen the user is looking at.
The model is trained on 48000 images of left eyes from the GazeCapure project.

A convolutional neural network (CNN) is implemented with 3 convolutional layers and 2 max pooling layers. The first 2 convolutional
layers have 32 filters each of size 5x5.The last convolutional layer has 64 filters each of size 3x3. After each convolutional layer, there is ReLu activation. 

Each pooling layer is max pooling of size 2x2 and stride of 2. No padding is implemented.

The optimizer used is AdamOptimizer with a learning rate of 0.001. 

The weights and biases were initialized using Xavier initialization. 

The loss function used for optimization is the mean Euclidean distance between the predicted coordinates and the labels.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Specifying file name containing data
npzfile = np.load("train_and_val.npz")

#Loading training data
train_eye_left = npzfile["train_eye_left"].astype('float32')
train_eye_right = npzfile["train_eye_right"].astype('float32')
train_face = npzfile["train_face"].astype('float32')
train_face_mask = npzfile["train_face_mask"]
train_y = npzfile["train_y"].astype('float32')
train_y = np.reshape(train_y,[48000,2])


#Loading validation data
val_eye_left = npzfile["val_eye_left"].astype('float32')
val_eye_right = npzfile["val_eye_right"].astype('float32')
val_face = npzfile["val_face"].astype('float32')
val_face_mask = npzfile["val_face_mask"]
val_y = npzfile["val_y"].astype('float32')
val_y = np.reshape(val_y,[5000,2])

print('Data Loaded')

    
#Scaling
L_train = train_eye_left/255.0

L_val = val_eye_left/255.0


print('Data Normalized')

val_shape = 10001
batch_size = 200
#val_batch_size = 500

num_filter1 = 32
num_filter2 = 64

filter1 = 5
filter2 = 3

mask_size = 25

#dimension of picture is 64x64
dim = 64

#Starting the graph
graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('input'):
    #initialize placeholders for input data
        tf_L_train = tf.placeholder(tf.float32, shape = (batch_size,dim,dim,3))
        #tf_R_train = tf.placeholder(tf.float32, shape = (batch_size,dim,dim,3))
        #tf_F_train = tf.placeholder(tf.float32, shape = (batch_size,dim,dim,3))
        #mask = tf.placeholder(tf.float32, [None,mask_size,mask_size], name = 'mask')
        labels = tf.placeholder(tf.float32, [batch_size,2], name = 'labels')
    
    with tf.name_scope('Validation_input'):
    #placeholders for validation data
        tf_L_val = tf.placeholder(tf.float32, shape = (batch_size,dim,dim,3))
        #tf_R_val = tf.placeholder(tf.float32, shape = (val_shape,dim,dim,3))
        #tf_F_val = tf.placeholder(tf.float32, shape = (val_shape,dim,dim,3))  
        val_labels = tf.placeholder(tf.float32, [batch_size,2], name = 'val_labels')
    
    #initialize weights
    
    #For Left Eye
    with tf.name_scope('Left_Eye_Weights'):
        w1_L = tf.get_variable("w1_L",[filter1,filter1,3,num_filter1],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w2_L = tf.get_variable("w2_L",[filter1,filter1,num_filter1,num_filter1],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w3_L = tf.get_variable("w3_L",[filter1,filter1,num_filter1,num_filter2],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w4_L = tf.get_variable("w4_L",[5184,2],initializer = tf.contrib.layers.xavier_initializer_conv2d())
    
    with tf.name_scope('Left_Eye_Biases'):
        b1_L = tf.get_variable("b1_L", [1,num_filter1],initializer = tf.constant_initializer(0.0))    
        b2_L = tf.get_variable("b2_L", [1,num_filter1],initializer = tf.constant_initializer(0.0))    
        b3_L = tf.get_variable("b3_L", [1,num_filter2],initializer = tf.constant_initializer(0.0))    
        b4_L = tf.get_variable("b4_L", [1,2],initializer = tf.constant_initializer(0.0))
    
    #Constructing the CNN
    def net(data):
        with tf.name_scope('Conv_1'):
            conv1 = tf.nn.conv2d(data,w1_L,[1,1,1,1],padding = 'VALID')
        with tf.name_scope('ReLu_1'):
            A1 = tf.nn.relu(conv1 + b1_L, name = 'A1')
        with tf.name_scope('Pool_1'):
            pool1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1] , padding = 'VALID')
        with tf.name_scope('Conv_2'):
            conv2 = tf.nn.conv2d(pool1,w2_L,[1,1,1,1],padding = 'VALID')
        with tf.name_scope('ReLu_2'):
            A2 = tf.nn.relu(conv2 + b2_L)
        with tf.name_scope('Pool_2'):
            pool2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID' )
        with tf.name_scope('Conv_3'):
            conv3 = tf.nn.conv2d(pool2,w3_L,[1,1,1,1],padding = 'VALID')
        with tf.name_scope('ReLu_3'):
            A3 = tf.nn.relu(conv3 + b3_L)
        #print(A3.get_shape())
        A3_shape = A3.get_shape().as_list()
        
        with tf.name_scope('ReLu_3_reshape'):
            A3_v = tf.reshape(A3,[A3_shape[0], A3_shape[1]*A3_shape[2]*A3_shape[3]])#vectorizing the last layer
        
                          
        return tf.matmul(A3_v,w4_L)+b4_L
    
    with tf.name_scope('Training_Prediction'):
        prediction = net(tf_L_train)
    with tf.name_scope('Validation_Prediction'):
        val_pred = net(tf_L_val)
    with tf.name_scope('Training_Error'):
        train_err = tf.reduce_mean(tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(prediction, labels)),reduction_indices=1)))
    with tf.name_scope('Validation_Error'):
        val_err = tf.reduce_mean(tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(val_pred, val_labels)),reduction_indices=1)))
    
    with tf.name_scope('Training_Operation'):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(train_err) 

train_error_list = []
val_error_list = []
i_list = []    
iterations = 10000
logs_path = "./graphs"   
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print('Variables Initialized')
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(iterations):
        #Creating batches
        off = (i * batch_size) % (train_y.shape[0] - batch_size)
        val_off = (i * batch_size) % (val_y.shape[0] - batch_size)
        batch_X = L_train[off:(off + batch_size), :, :, :]
        batch_Y = train_y[off:(off + batch_size),:]
        val_batch_X = L_val[val_off:(val_off + batch_size), :, :, :]
        val_batch_Y = val_y[val_off:(val_off + batch_size),:]
        
        #Feeding in the data into the placeholders                          
        feed_dict = {tf_L_train : batch_X, labels : batch_Y}
        _,l,predictions = session.run([optimizer, train_err,prediction],feed_dict = feed_dict)
    
        #print(predictions)
        if(i%100 == 0):
            val_dict = {tf_L_val: val_batch_X, val_labels : val_batch_Y}
            val_error = session.run(val_err, feed_dict = val_dict)
            train_error_list.append(l)
            val_error_list.append(val_error)
            i_list.append(i)
            print('Training error at iteration', i, '=', l)
            print('Validation error at iteration',i, '=', val_error)
    
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.plot(i_list,train_error_list,'b',label = 'Training error')
    ax1.plot(i_list,val_error_list,'r',label = 'Validation error')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Error(cm)')  
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.095),fancybox=True, shadow=True, ncol=3)
    fig.tight_layout()
    plt.show()

writer.close()