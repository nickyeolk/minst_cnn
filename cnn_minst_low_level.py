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

mnist=tf.contrib.learn.datasets.load_dataset('mnist')

def conv_layer(inputt,size_in,size_out,name='conv'):
    with tf.name_scope(name):
        w=tf.Variable(tf.truncated_normal([5,5,size_in,size_out],stddev=0.1),name='W')
        b=tf.Variable(tf.constant(0.0,shape=[size_out]),name='B')
        conv=tf.nn.conv2d(inputt,w,strides=[1,1,1,1],padding='SAME')
        act=tf.nn.relu(conv+b)
        w_trans=tf.transpose(w,[2,3,0,1])
        w_trans1=tf.reshape(w_trans,[size_in,size_out,5,5,1])
        tf.summary.image('weights',w_trans1[0,:,:,:],size_out)
        tf.summary.histogram('weights',w)
        tf.summary.histogram('biases',b)
        tf.summary.histogram('activations',act)
        return tf.nn.max_pool(act,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def fc_layer(inputt,size_in,size_out,name='fc'):
    with tf.name_scope(name):
        w=tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name='W')
        b=tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act=tf.nn.relu(tf.matmul(inputt,w)+b)
        tf.summary.histogram('weights',w)
        tf.summary.histogram('biases',b)
        tf.summary.histogram('activations',act)
        return act

def minst_model(learning_rate,use_two_conv,use_two_fc,hparam):
    tf.reset_default_graph()
    sess=tf.Session()
    
    #setup placeholders, reshape data
    x=tf.placeholder(tf.float32,shape=[None,784],name='x')
    x_image=tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',x_image,4)
    y=tf.placeholder(tf.float32,shape=[None,10],name='labels')
    
    if(use_two_conv):
        conv1=conv_layer(x_image,1,32,'conv1')
        conv_out=conv_layer(conv1,32,64,'conv2')
    else:
        conv1=conv_layer(x_image,1,64,'conv1')
        conv_out=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    flattened=tf.reshape(conv_out,[-1,7*7*64])
    
    if(use_two_conv):
        fc1=fc_layer(flattened,7*7*64,1024,'fc1')
        embedding_input=fc1
        embedding_size=1024
        logits=fc_layer(fc1,1024,10,'fc2')
    else:
        logits=fc_layer(flattened,7*7*64,10,'fc')
    
    with tf.name_scope('xent'):
        xent=tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,labels=y),name='xent')
        tf.summary.scalar('xent',xent)
        
    with tf.name_scope('train'):
        #minimize combines compute and combine gradients, maybe can call
        #explicitly to check gradient backprop
        train_step=tf.train.AdamOptimizer(learning_rate).minimize(xent)
    
    with tf.name_scope('accuracy'):
        correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
        
    summ=tf.summary.merge_all()
    
    #some embedding and saver stuff here
    
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('./log_mnist_low')
    writer.add_graph(sess.graph)
    
    #more embeddings stuff
    
    for i in range(2001):
        batch=mnist.train.next_batch(100)
        hot_labels=tf.one_hot(batch[1],10)
        if i%5==0:
            [train_accuracy,s]=sess.run([accuracy,summ],feed_dict={x:batch[0],\
                y:hot_labels.eval(session=sess)})#this has to be one hot encoded!!!
            writer.add_summary(s,i)
        #model backup here
        sess.run(train_step,feed_dict={x:batch[0],y:hot_labels.eval(session=sess)})
        
    #function for making parameter string for testing hyper param
        
def main():
        for learning_rate in [1e-4]:
            
            for use_two_fc in [True]:
                for use_two_conv in [True]:
                    #make param string
                    print('starting run')
                    minst_model(learning_rate,use_two_fc,use_two_conv,'test_run')

if __name__=='__main__':
    main()
