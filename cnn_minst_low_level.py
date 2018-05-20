# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:16:05 2018
Testing out using CNNs on the MNIST dataset
This time not using estimators, but building everything from scratch.
lookin to fully utilize tensorboard as well.
scalars, histograms, distributions
this time looking to add graphs and embeddings too
@author: likkhian
"""
import tensorflow as tf
import numpy as np

def conv_layer(inputt,size_in,size_out,name='conv'):
    with tf.name_scope(name):
        w=tf.Variable(tf.truncated_normal([5,5,size_in,size_out],stddev=0.1),name='W')
        b=tf.Variable(tf.constant(0,shape=[size_out]),name='B')
        conv=tf.nn.conv2d(inputt,w,strides=[1,1,1,1],padding='SAME')
        act=tf.nn.relu(conv+b)
        tf.summary.histogram('weights',w)
        tf.summary.histogram('biases',b)
        tf.summary.histogram('activations',act)
        return tf.nn.max_pool(act,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def fc_layer(inputt,size_in,size_out,name='fc'):
    with tf.name_scope(name):
        w=tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name='W')
        b=tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act=tf.relu(tf.matmul(inputt,w)+b)
        tf.summary.histogram('weights',w)
        tf.summary.historgram('biases',b)
        tf.summary.histogram('activations',act)
        return act

