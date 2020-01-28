# -*- coding: utf-8 -*-
"""
Pranesh Navarathna
ECSE 6965 Deep Learning
Final Project - Eye Gaze Position Estimation

Implemented in Tensorflow v1.0.1

This model takes in pictues of left eyes and right eyes and predicts what position on a phone screen the user is looking at.
The model is trained on 48000 images of left and right eyes from the GazeCapure project

A convolutional neural network (CNN) is implemented with 3 convolutional layers and 2 max pooling layers. The first 2 convolutional
layers have 32 filtes each of size 5x5.The last convolutional layer has 64 filters each of size 3x3. After each convolutional layer, there is ReLu activation. 

Each pooling layer is max pooling of size 2x2 and stride of 2. No padding is implemented. 

The outputs of the CNN for left and right eyes are concatenated and fed into a feedforward neural network with 2 ReLu activated hidden layers of 20 nodes each. 
The output nodes of the feedforward network have linear activation for regression purposes.

The optimizer used is AdamOptimizer with a learning rate of 0.001. 

The weights and biases were initialized using Xavier initialization. 

The loss function used for optimization is the mean Euclidean distance between the predicted coordinates and the labels.

The model is seen to achieve a batch validation error of 1.74 cm and an overall valdiation error of 2.05 cm
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

    
#Scaling/Normalizing
#Normalizing Training Data
L_train = train_eye_left/255.0
R_train = train_eye_right/255.0
#Normalizing Validation Data
L_val = val_eye_left/255.0
R_val = val_eye_right/255.0

print('Data Normalized')

#Specifying training batch size and validation batch size
batch_size = 100#training batch size
val_batch_size = 100#validation batch size

#Defining the number of filters to be used 
num_filter1 = 32
num_filter2 = 64

#Defining the size of the filters
filter1 = 5
filter2 = 3
#Defining the mask size for the placeholder
mask_size = 25
#Defining size of the hidden layer for the feedforward network
hidden_size = 20

#dimension of picture is 64x64
dim = 64

#Starting the graph
graph = tf.Graph()

with graph.as_default():
    
    with tf.name_scope('input'):
    #initialize placeholders for input data
        tf_L_train = tf.placeholder(tf.float32, shape = (batch_size,dim,dim,3))
        tf_R_train = tf.placeholder(tf.float32, shape = (batch_size,dim,dim,3))
        tf_F_train = tf.placeholder(tf.float32, shape = (batch_size,dim,dim,3))
        mask = tf.placeholder(tf.float32, [None,mask_size,mask_size], name = 'mask')
        labels = tf.placeholder(tf.float32, [batch_size,2], name = 'labels')
        
    with tf.name_scope('Validation_input'):
        #placeholders for validation data
        tf_L_val = tf.placeholder(tf.float32, shape = (val_batch_size,dim,dim,3))
        tf_R_val = tf.placeholder(tf.float32, shape = (val_batch_size,dim,dim,3))
        #tf_F_val = tf.placeholder(tf.float32, shape = (val_shape,dim,dim,3))  
        val_labels = tf.placeholder(tf.float32, [val_batch_size,2], name = 'val_labels')
    
    #initialize weights
    
    with tf.name_scope('Left_Eye_Weights'):
        #Left Eye Weights
        w1_L = tf.get_variable("w1_L",[filter1,filter1,3,num_filter1],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w2_L = tf.get_variable("w2_L",[filter1,filter1,num_filter1,num_filter1],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w3_L = tf.get_variable("w3_L",[filter1,filter1,num_filter1,num_filter2],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w4_L = tf.get_variable("w4_L",[5184,2],initializer = tf.contrib.layers.xavier_initializer_conv2d())
    
    with tf.name_scope('Left_Eye_Biases'):
        #Left Eye Biases
        b1_L = tf.get_variable("b1_L", [1,num_filter1],initializer = tf.constant_initializer(0.0))   
        b2_L = tf.get_variable("b2_L", [1,num_filter1],initializer = tf.constant_initializer(0.0))   
        b3_L = tf.get_variable("b3_L", [1,num_filter2],initializer = tf.constant_initializer(0.0))    
        b4_L = tf.get_variable("b4_L", [1,2],initializer = tf.constant_initializer(0.0))
    
    with tf.name_scope('Right_Eye_Weights'):
        #Right Eye Weights
        w1_R = tf.get_variable("w1_R",[filter1,filter1,3,num_filter1],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w2_R = tf.get_variable("w2_LR",[filter1,filter1,num_filter1,num_filter1],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w3_R = tf.get_variable("w3_R",[filter1,filter1,num_filter1,num_filter2],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w4_R = tf.get_variable("w4_R",[5184,2],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        #Right Eye Biases
    with tf.name_scope('Right_Eye_Biases'):
        b1_R = tf.get_variable("b1_R", [1,num_filter1],initializer = tf.constant_initializer(0.0))    
        b2_R = tf.get_variable("b2_R", [1,num_filter1],initializer = tf.constant_initializer(0.0))    
        b3_R = tf.get_variable("b3_R", [1,num_filter2],initializer = tf.constant_initializer(0.0))    
        b4_R = tf.get_variable("b4_R", [1,2],initializer = tf.constant_initializer(0.0))
    
    # net() is the CNN 
    def net(data,w1,b1,w2,b2,w3,b3,w4,b4):
        with tf.name_scope('Conv_1'): 
            conv1 = tf.nn.conv2d(data,w1,[1,1,1,1],padding = 'VALID')
        with tf.name_scope('ReLu_1'):
            A1 = tf.nn.relu(conv1 + b1, name = 'A1')
        with tf.name_scope('Pool_1'):
            pool1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1] , padding = 'VALID')
        
        with tf.name_scope('Conv_2'):
            conv2 = tf.nn.conv2d(pool1,w2,[1,1,1,1],padding = 'VALID')
        with tf.name_scope('ReLu_2'):
            A2 = tf.nn.relu(conv2 + b2)
        with tf.name_scope('Pool_2'):
            pool2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID' )
        with tf.name_scope('Conv_3'):
            conv3 = tf.nn.conv2d(pool2,w3,[1,1,1,1],padding = 'VALID')
        with tf.name_scope('ReLu_3'):
            A3 = tf.nn.relu(conv3 + b3)
        
        A3_shape = A3.get_shape().as_list()#getting the shape to reshape
        
        with tf.name_scope('ReLu_3_reshape'):
            A3_v = tf.reshape(A3,[A3_shape[0], A3_shape[1]*A3_shape[2]*A3_shape[3]])#vectorizing the last layer
        
                          
        return tf.matmul(A3_v,w4_L)+b4
    
    #Calling the CNN on the left eye data
    with tf.name_scope('Left_Eye_Training_Output'):
        left_prediction = net(tf_L_train,w1_L,b1_L,w2_L,b2_L,w3_L,b3_L,w4_L,b4_L)
    #Calling the CNN on the right eye data
    with tf.name_scope('Right_Eye_Training_Output'):
        right_prediction = net(tf_R_train,w1_R,b1_R,w2_R,b2_R,w3_R,b3_R,w4_R,b4_R)
    #Calling CNN for validation
    with tf.name_scope('Left_Eye_Validation_Output'):
        left_val = net(tf_L_val,w1_L,b1_L,w2_L,b2_L,w3_L,b3_L,w4_L,b4_L)
        
    with tf.name_scope('Right_Eye_Validation_Output'):
        right_val = net(tf_R_val,w1_R,b1_R,w2_R,b2_R,w3_R,b3_R,w4_R,b4_R)
        
    #Concatenating the outputs of left eye and right eye
    with tf.name_scope('Training_Eyes_Concatenate'):
        train_eyes = tf.concat([left_prediction,right_prediction],axis=1)
        
    with tf.name_scope('Validation_Eyes_Concatenate'):
        val_eyes = tf.concat([left_val,right_val],axis=1)
    
    #Creating a fully connected network with 2 hidden layers
    #Defining weights
    with tf.name_scope('Fully_Connected_Layer_Weights'):
        w_H1 = tf.get_variable("w_H1",[4,hidden_size],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w_H2 = tf.get_variable("w_H2",[hidden_size,hidden_size],initializer = tf.contrib.layers.xavier_initializer_conv2d())
        w_out = tf.get_variable("w_out",[hidden_size,2],initializer = tf.contrib.layers.xavier_initializer_conv2d())
    #Biases
    with tf.name_scope('Fully_Connected_Layer_Biases'):
        b_H1 = tf.get_variable("b_H1", [1,hidden_size],initializer = tf.constant_initializer(0.0))    
        b_H2 = tf.get_variable("b_H2", [1,hidden_size],initializer = tf.constant_initializer(0.0))    
        b_out = tf.get_variable("b_out",[1,2],initializer = tf.constant_initializer(0.0))
    
    #Feed Forward Neural Net
    def fully_connected(sample):
        with tf.name_scope('Hidden_1'):
            H1 = tf.nn.relu(tf.matmul(sample,w_H1) + b_H1)
        with tf.name_scope('Hidden_2'):
            H2 = tf.nn.relu(tf.matmul(H1,w_H2) + b_H2)
        with tf.name_scope('Fully_Connected_Output'):
            out = tf.matmul(H2,w_out) + b_out
        
        return out
        
    #Runningt the feedforward network on the combined eye data    
    with tf.name_scope('Training_Prediction'):
        prediction = fully_connected(train_eyes)
    
    with tf.name_scope('Validation_Prediction'):
        val_pred = fully_connected(val_eyes)
        
    #Using mean Euclidean distance as error
    with tf.name_scope('Training_Error'):
        train_err = tf.reduce_mean(tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(prediction, labels)),reduction_indices=1)))
    
    with tf.name_scope('Validation_Error'):
        val_err = tf.reduce_mean(tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(val_pred, val_labels)),reduction_indices=1)))
    
    #Defining the training operation
    with tf.name_scope('Training_Operation'):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(train_err) 
        
    #Saving to a collection
    # Create the collection.
#    tf.get_collection("validation_nodes")
#    # Add stuff to the collection.
#    tf.add_to_collection("validation_nodes", tf_L_train)
#    tf.add_to_collection("validation_nodes", tf_R_train)
#    tf.add_to_collection("validation_nodes", tf_F_train)
#    tf.add_to_collection("validation_nodes", mask)
#    tf.add_to_collection("validation_nodes", prediction)
        

#initalizing lists and iterations for the training loop    
train_error_list = []
val_error_list = []
i_list = []    
iterations = 20001

#defining where to save the graphs for Tensorboard visualization
logs_path = "./graphs2" 
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print('Variables Initialized')
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())#write the graph elements to the graph
    
    for i in range(iterations):
        #Creating batches
        off = (i * batch_size) % (train_y.shape[0] - batch_size)
        val_off = (i * val_batch_size) % (val_y.shape[0] - val_batch_size)
        batch_X_L = L_train[off:(off + batch_size), :, :, :]
        batch_X_R = R_train[off:(off + batch_size), :, :, :]
        batch_Y = train_y[off:(off + batch_size),:]
        val_batch_X_L = L_val[val_off:(val_off + val_batch_size), :, :, :]
        val_batch_X_R = R_val[val_off:(val_off + val_batch_size), :, :, :]
        val_batch_Y = val_y[val_off:(val_off + val_batch_size),:]
        
        #Feeding in the data into the placeholders                          
        feed_dict = {tf_L_train : batch_X_L, tf_R_train : batch_X_R, labels : batch_Y}
        _,l = session.run([optimizer, train_err],feed_dict = feed_dict)
    
        #Run the validation data every 100 iterations
        if(i%100 == 0):
            val_dict = {tf_L_val: val_batch_X_L,tf_R_val : val_batch_X_R, val_labels : val_batch_Y}
            val_error = session.run(val_err, feed_dict = val_dict)
            train_error_list.append(l)
            val_error_list.append(val_error)
            i_list.append(i)
            print('Training error at iteration', i, '=', l)
            print('Validation error at iteration',i, '=', val_error)
            #It was observed that the lowest error that could be achieved was below 1.8 cm, so the loop breaks as soon as a validation
            #error of below 1.8 cm is reached
            if val_error < 1.8:
                break
    
    #Saving the model    
    #saver = tf.train.Saver()
    #save_path = saver.save(session, "./my-model")
    
    #Plotting the errors
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.plot(i_list,train_error_list,'b',label = 'Training error')
    ax1.plot(i_list,val_error_list,'r',label = 'Validation error')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Error(cm)')  
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.095),fancybox=True, shadow=True, ncol=3)
    fig.tight_layout()
    plt.show()

writer.close()#closing the writer for the Tensorboard graph