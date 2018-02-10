from __future__ import absolute_import
from __future__ import print_function

"""handwriten_digits.py: Basic nueral networks for identifying handwriiten digits using tensorFlow
Activation: sigmoid and softmax
Accuracy acheived  :  97.24 %. training_steps = 100000

"""


__author__ = "Vishal Jasrotia"
__date__   = "Feb, 06, 2018"
__organization__ = "Stony Brook Univerity, NY, USA"
__email__ = "jasrotia.vishal@stonybrook.edu"
__version__ = "0.1"


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data



learning_rate = 0.0002
batch_size = 100
pic_number = 10   # neurals count
pic_size = 28
training_steps = 1000000
print_iter = 10
collect_interval = 100



layer_1 = 4
layer_2 = 8
layer_3 = 12
layer_4 = 200



# W0 = tf.Variable(tf.truncated_normal([pic_size*pic_size, layer_0], stddev = 0.1))
# B0 = tf.Variable(tf.zeros([layer_0]))

#                                      #filter size   #input channel   #output channel
W1 = tf.Variable(tf.truncated_normal([   5,    5,      1,               layer_1], stddev = 0.1))
B1 = tf.Variable(tf.ones([layer_1])/10)
 
 


W2 = tf.Variable(tf.truncated_normal([ 5,    5,      layer_1,               layer_2], stddev = 0.1))
B2 = tf.Variable(tf.ones([layer_2])/10)


W3 = tf.Variable(tf.truncated_normal([ 4,    4,      layer_2,               layer_3], stddev = 0.1))
B3 = tf.Variable(tf.ones([layer_3])/10)



W4 = tf.Variable(tf.truncated_normal([7*7*layer_3, layer_4], stddev = 0.1))
B4 = tf.Variable(tf.ones([layer_4])/10)

W5 = tf.Variable(tf.truncated_normal([layer_4, pic_number], stddev = 0.1))
B5 = tf.Variable(tf.zeros([pic_number])/10)

start  =   int(time.time())
 
#mnist data from handwriten digits

dataset_name = "MNIST_data"
mnist = input_data.read_data_sets(dataset_name+"/", one_hot=True)
   
   

#model with diff weight and biases
x = tf.placeholder(tf.float32, [None, pic_size*pic_size])

#Y0 = tf.sigmoid(tf.matmul(x,W0) + B0)

Y1 = tf.sigmoid(tf.matmul(x,W1) + B1)

Y2 = tf.sigmoid(tf.matmul(Y1,W2) + B2)
Y3 = tf.sigmoid(tf.matmul(Y2,W3) + B3)
Y4 = tf.sigmoid(tf.matmul(Y3,W4) + B4)
y = tf.nn.softmax(tf.matmul(Y4,W5) + B5)

#W = tf.Variable(tf.zeros([pic_size*pic_size, pic_number]))
#b = tf.Variable(tf.zeros([pic_number]))

#y = tf.nn.softmax(tf.matmul(x, W) + b)

#loss function and optimizer
y_ = tf.placeholder(tf.float32, [None, pic_number])
cross_entropy =-tf.reduce_sum(y_*tf.log(y)) # can be mean also
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

  

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#train_step = optimizer.minimize(cross_entropy)
   
# correct answers
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
   
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
   
   
x_vals_train = []
y_vals_train = []
x_vals_test = []
y_vals_test = []
 
#train model
test_data = {x :mnist.test.images,y_:mnist.test.labels}
with tf.Session() as sess:
    sess.run(init_op)
    for i in xrange(training_steps+1):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
         
        if i%print_iter == 0:
            accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
             
            x_vals_train.append(i)
            y_vals_train.append(accuracy_val.mean()*100)
            
            
            accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=test_data)
            x_vals_test.append(i)
            y_vals_test.append(accuracy_val.mean()*100) 
             
            #print("************************")
            #print(dir(accuracy_val))
            #print("************************")
            #print(dir(loss_val))
         
            print("Iteration :  {} , Loss : {:.5} - {:.5} Accuracy :  {:.5} - {:.5}".format(i,loss_val.min() , loss_val.max(), accuracy_val.min()*100, accuracy_val.max()*100))
               
       
    test_data = {x :mnist.test.images,y_:mnist.test.labels}
    accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=test_data)
    print("Result on test data")
    print("Loss : {:.5} - {:.5} Accuracy :  {:.5} - {:.5}".format(loss_val.min() , loss_val.max(), accuracy_val.min()*100, accuracy_val.max()*100))
    print(dir(loss_val))
    save_path = saver.save(sess, "./output/model.ckpt")
       
    print ("Model saved in file: ", save_path)


fig = plt.figure(num=None, figsize=(16, 9), dpi=360, facecolor='w', edgecolor='k')

fig.suptitle("TensorFlow Accuracy graph for mnist.")


linecos1 = plt.plot(x_vals_train, y_vals_train, 'r-', label='Accuracy (Train data)')
linecos2 = plt.plot(x_vals_test, y_vals_test, 'b-', label='Accuracy (Test data)')


plt.xlabel("Iteration")
plt.ylabel("Accuracy ( Percentage %)")


plt.ylim(0,100)
x_limit = max(max(max(x_vals_train), max(x_vals_test)), 105)
plt.xlim(0,x_limit)
interval = x_limit/20.0
plt.xticks(np.arange(0, x_limit, interval))
plt.yticks(np.arange(0, 105, 5.0))

#print(x_vals)
#print(y_vals)
#plt.figtext(0.20, 0.08, "TensorFlow Accuracy of batch of 100 set", horizontalalignment='right') 
plt.figtext(0.28, 0.06, "Accuracy on test data set  : {:.3} percent.".format(accuracy_val.mean()*100), horizontalalignment='right') 
plt.figtext(0.195, 0.04, "Total loss:  %s"%(loss_val.mean()),horizontalalignment='right') 
plt.figtext(0.23, 0.02, "Number of training steps :  %s"%(training_steps),horizontalalignment='right') 


plt.minorticks_on()
plt.legend()

plt.grid()
#plt.show()
stamp = "_".join(time.asctime().split())
file_name = "./output/tensor_accuracy_"+os.path.basename(__file__).replace(".py" , "_")+dataset_name+"_"+stamp+".png"

print("graph saved in : ", file_name)
plt.savefig(file_name)


