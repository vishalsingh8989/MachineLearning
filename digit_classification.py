from __future__ import absolute_import
from __future__ import print_function

"""handwriten_digits.py: Basic nueral networks for identifying handwriiten digits using tensorFlow
Accuracy acheived  :  92 %.

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



learning_rate = 0.002
batch_size = 100
pic_number = 10   # neurals count
pic_size = 28
training_steps = 200
print_iter = 1


 
start  =   int(time.time())
 
#mnist data from handwriten digits

dataset_name = "MNIST_data"
mnist = input_data.read_data_sets(dataset_name+"/", one_hot=True)
   
#model
x = tf.placeholder(tf.float32, [None, pic_size*pic_size])
W = tf.Variable(tf.zeros([pic_size*pic_size, pic_number]))
b = tf.Variable(tf.zeros([pic_number]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#loss function and optimizer
y_ = tf.placeholder(tf.float32, [None, pic_number])
cross_entropy =-tf.reduce_sum(y_*tf.log(y)) # can be mean also
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
   
   
# correct answers
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
   
init_op = tf.initialize_all_variables()

saver = tf.train.Saver()
   
   
x_vals = []
y_vals = []
 
#train model
with tf.Session() as sess:
    sess.run(init_op)
    for i in xrange(training_steps+1):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
         
        if i%print_iter == 0:
            accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
             
            x_vals.append(i)
            y_vals.append(accuracy_val.mean()*100)
             
             
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


linecos = plt.plot(x_vals, y_vals, 'r-', label='Accuracy')


plt.xlabel("Iteration")
plt.ylabel("Accuracy ( Percentage %)")


plt.ylim(0,100)
x_limit = max(max(x_vals), 105)
plt.xlim(0,x_limit)
interval = x_limit/20.0
plt.xticks(np.arange(0, x_limit, interval))
plt.yticks(np.arange(0, 105, 5.0))

#print(x_vals)
#print(y_vals)
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


