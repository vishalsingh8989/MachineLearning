from __future__ import absolute_import
from __future__ import print_function

"""handwriten_digits.py: Basic nueral networks for identifying handwriiten digits using tensorFlow"""
from tensorflow.contrib import training

__author__ = "Vishal Jasrotia"
__date__   = "Feb, 06, 2018"
__organization__ = "Stony Brook Univerity, NY, USA"
__email__ = "jasrotia.vishal@stonybrook.edu"
__version__ = "0.1"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


learning_rate = 0.0001
batch_size = 100
pic_number = 10
pic_size = 28
training_steps = 10000000

print_iter = 1000

#mnist data from handwriten digits
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#model
x = tf.placeholder(tf.float32, [None, pic_size*pic_size])
W = tf.Variable(tf.zeros([pic_size*pic_size, pic_number]))
b = tf.Variable(tf.zeros([pic_number]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

#loss function and optimizer
y_ = tf.placeholder(tf.float32, [None, pic_number])
cross_entropy =tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y))) # can be mean also
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


# # correct answers
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()



#train model
with tf.Session() as sess:
    sess.run(init_op)
    for i in xrange(training_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i%print_iter == 0:
            accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
            #print("************************")
            #print(dir(accuracy_val))
            #print("************************")
            #print(dir(loss_val))
            print("Loss : {:.5} - {:.5} Accuracy :  {:.5} - {:.5}".format(loss_val.min() , loss_val.max(), accuracy_val.min()*100, accuracy_val.max()*100))
            
    
    test_data = {x :mnist.test.images,y_:mnist.test.labels}
    accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=test_data)
    print("Loss : {:.5} - {:.5} Accuracy :  {:.5} - {:.5}".format(loss_val.min() , loss_val.max(), accuracy_val.min()*100, accuracy_val.max()*100))
    save_path = saver.save(sess, "./output/model.ckpt")
    
    print ("Model saved in file: ", save_path)
