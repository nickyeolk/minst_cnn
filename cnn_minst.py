# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:17:27 2018
Just testing out using CNNs on the MINST dataset
This script demonstrates how to 
Use tf.layers to construct my own estimators.
Output log files using tf.summary for viewing in tensorboard
- scalars, images, histograms
Output visualizations of the convolutional filters
note: Estimators automatically output log files in the model directory.
@author: likkhian
"""
import numpy as np
import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)
    
def cnn_model_fn(features,labels,mode):
    '''model function for cnn'''
    #input layer
    input_layer=tf.reshape(features['x'],[-1,28,28,1])
    
    #convolutional layer 1, including namescopes. Examine conv2d vs Conv2d!
    #Conv2d is a class. conv2d is a function that uses the Conv2d class.
    with tf.name_scope('lik_conv1'):
        conv1=tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu,
            name='conv1')
        conv1_kernel=tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]
        conv1_kernel_transposed=tf.transpose(conv1_kernel,[3,0,1,2])
        conv1_bias = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/bias')[0]
        convimg1=tf.reshape(conv1,[-1,28,28,32,1])
        convimg2=tf.transpose(convimg1,[0,3,1,2,4])
        
    
    #pooling layer 1
    pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)
    
    #convolutional layer 2, and pooling layer 2
    with tf.name_scope('lik_conv2'):
        conv2=tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu)
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
    
    #dense layer
    pool2_flat=tf.reshape(pool2,[-1,7*7*64])
    dense=tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)
    #extract weights and bias for tensorboard histogram
    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(dense.name)[0] + '/kernel:0')
    bias = tf.get_default_graph().get_tensor_by_name(os.path.split(dense.name)[0] + '/bias:0')
    dropout=tf.layers.dropout(inputs=dense,rate=0.4,
                              training=mode==tf.estimator.ModeKeys.TRAIN)
    
    #logits layer
    logits=tf.layers.dense(inputs=dropout,units=10)
    
    predictions={
        #generate predictions (for PREDICT and EVAL mode)
        'classes':tf.argmax(input=logits,axis=1),
        #Add softmax_tensor to the graph. used for predict and logging hook
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor')}
        
    if(mode==tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
        
    #calculate loss
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    
    #save loss as a scalar
    tf.summary.scalar('lik_loss',loss)
    tf.summary.image('lik_input',input_layer,4)
    tf.summary.image('conv1_filter',conv1_kernel_transposed,32)
    tf.summary.histogram('conv1_bias',conv1_bias)
    tf.summary.histogram('lik_denasa_wts',weights)
    tf.summary.histogram('lik_dense_bias',bias)
    tf.summary.image('lik_convimg',convimg2[0,:,:,:],32)
    #put in summary so it will show up in training
#    accuracy=tf.metrics.accuracy(labels=labels,predictions=predictions['classes'])
#    tf.summary.scalar('lik_acc',accuracy[1])

    #add evaluation metrics. Moved it here so accuracy will be in training too
    eval_metric_ops={
        'accuracy':tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])}
    tf.summary.scalar('lik_acc',eval_metric_ops['accuracy'][1])

    #configure training op
    if(mode==tf.estimator.ModeKeys.TRAIN):
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op=optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    #load training and eval data
    mnist=tf.contrib.learn.datasets.load_dataset('mnist')
    train_data=mnist.train.images
    train_labels=np.asarray(mnist.train.labels,dtype=np.int32)
    eval_data=mnist.train.images
    eval_labels=np.asarray(mnist.train.labels,dtype=np.int32)
    #create the estimator
    mnist_classifier=tf.estimator.Estimator(model_fn=cnn_model_fn,
                                            model_dir='./mnist_convnet_model2')
    #set up logging
    tensors_to_log={'probabilities':'softmax_tensor'}
    logging_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                            every_n_iter=100)
    #training time
    train_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x':train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,steps=20000,hooks=[logging_hook])
    eval_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x':eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results=mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__=='__main__':
    tf.app.run()