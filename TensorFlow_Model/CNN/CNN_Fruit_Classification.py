#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
import os
import matplotlib.pyplot as plt


# In[2]:


trainlist, testlist = [], []
with open('train.txt') as f:
    for line in f:
        tmp = line.strip().split()
        trainlist.append([tmp[0], tmp[1]])
        
with open('test.txt') as f:
    for line in f:
        tmp = line.strip().split()
        testlist.append([tmp[0], tmp[1]])


# In[3]:


IMG_H = 100
IMG_W = 100
IMG_C = 3

def readimg(path):
    img = plt.imread(path)
    return img

def batch(path, batch_size):
    img, label, paths = [], [], []
    for i in range(batch_size):
        img.append(readimg(path[0][0]))
        label.append(int(path[0][1]))
        path.append(path.pop(0))
        
    return img, label


# In[14]:


num_class = 3 

with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
    Y = tf.placeholder(tf.int32, [None])
    
    with tf.variable_scope('CNN'):
        net = tf.layers.conv2d(X, 20, 3, (2, 2), padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.conv2d(net, 40, 3, (2, 2), padding='same', activation=tf.nn.relu)
        net = tf.layers.flatten(net)
     
        out = tf.layers.dense(net, num_class)
        
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= Y, logits=out))
        
    train = tf.train.AdamOptimizer(1e-3).minimize(loss)
    saver = tf.train.Saver()


# In[18]:


np.sum([np.product(var.shape) for var in g.get_collection('trainable_variables')]).value


# In[16]:


batch_size = 1461
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30):
        batch_data, batch_label = batch(trainlist, batch_size)
        _, l = sess.run([train, loss], feed_dict = {X: batch_data, Y: batch_label})
        print(i, l)
        
    saver.save(sess, 'logs/model.ckpt', global_step = i+1)


# In[17]:


acc = 0
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint('logs')
    if checkpoint:
        saver.restore(sess, checkpoint)
    for i in range(len(testlist)):
        batch_data, batch_label = batch(testlist, 1)
        logit = sess.run(out, feed_dict = {X:batch_data})
        if np.argmax(logit[0]) == batch_label[0]:
            acc += 1
        else:
            print(logit[0], batch_label[0])
            
    print(acc/len(testlist))


# In[ ]:




