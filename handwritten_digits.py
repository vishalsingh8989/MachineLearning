from __future__ import absolute_import
from __future__ import print_function

"""handwriten_digits.py: Basic nueral networks for identifying handwriiten digits using tensorFlow"""

__author__ = "Vishal Jasrotia"
__date__   = "Feb, 06, 2018"
__organization__ = "Stony Brook Univerity, NY, USA"
__email__ = "jasrotia.vishal@stonybrook.edu"
__version__ = "0.1"




import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


learning_rate = 0.003
batch_size = 100
pic_number = 10
pic_size = 28

from tensorflow.examples.tutorials.mnist import input_data

#handwrittens number with labels
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 
# placeholder_x = tf.placeholder(tf.float32, [None, pic_size*pic_size])
# W = tf.Variable(tf.zeros([pic_size*pic_size,pic_number]))
# b = tf.Variable(tf.zeros([pic_number]))
# 
# init = tf.global_variables_initializer()
# 
# 
# #flatten the image
# 
# placeholder_y = tf.nn.softmax(tf.matmul(tf.reshape(placeholder_x, [-1, pic_size*pic_size]), W) + b)
# Y_ = tf.placeholder(tf.float32, [None, pic_number])
# 
# 
# #loss function 
# cross_entropy = -tf.reduce_sum(Y_*tf.log(placeholder_y)) 
# 
# 
# # correct answers
# is_correct = tf.equal(tf.argmax(placeholder_y,1), tf.argmax(Y_, 1))
# accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
# 
# #optimizer and train
# optimizer = tf.train.GradientDescentOptimizer(learning_rate) #size is  = W size =  [784,10] =  7840 . Partial derivatives, Formal derivatives
# train_step = optimizer.minimize(cross_entropy)



# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)


# # correct answers
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#train model over 1000 iterations

for i in range(1000):
    #train this model by feeding batcn x and correct labels
    x_batch , y_batch = mnist.train.next_batch(batch_size)
    feed_train = {x:x_batch , y: y_batch}
    sess.run(train_step , feed_dict=feed_train)
    
    #for checking
    accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=feed_train)
    
    print("loss mean : " , loss_val.mean())
    #test with test data and correct lables for accuracy and loss in every run to check  convergence.
    test_data = {x :mnist.test.images,y_:mnist.test.labels}
    accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=test_data)
    
    
    
    
    
    













