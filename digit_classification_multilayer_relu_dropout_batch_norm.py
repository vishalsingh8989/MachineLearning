from __future__ import absolute_import
from __future__ import print_function

"""handwriten_digits.py: Basic nueral networks for identifying handwriiten digits using tensorFlow
Activation: relu and softmax with dropout
Accuracy acheived  :  97.24 %. training_steps = 100000

Test result : 
 ____________________________________________________________________________________________________________________________________________________
|     Activation function             |  Number of layer   | Learning fxn |  Learning_rate  |  Batch_size |  Training_steps | Accuracy |   Total loss | 
|  sigmoid and softmax               |  4                 | GDO          |  0.0005         |   100       |    100000       |  96.57   |    1559.6    | 
|  sigmoid and softmax               |  4                 | GDO          |  0.0002         |   100       |    200000       |          |


"""

__author__ = "Vishal Jasrotia"
__date__   = "Feb, 12, 2018"
__organization__ = "Stony Brook Univerity, NY, USA"
__email__ = "jasrotia.vishal@stonybrook.edu"
__version__ = "0.1"


import os
import csv
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from testresults import update_results


start = time.time()

activation_fxn = "relu and softmax with decaying learning rate"
nums_of_layer = 4
learning_fxn = "AdamOptimizer"
dataset_type = "MNIST"
dataset_name = "MNIST_data"



max_learning_rate = 0.003
min_learning_rate = 0.00001
decay_speed = 2000.0
learning_rate = 0.0005
batch_size = 100
pic_number = 10   # neurals count
pic_size = 28
training_steps = 3000
print_iter = 100
collect_interval = 10

update_test_result = True

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


saver = tf.train.Saver()
# init
init_op = tf.global_variables_initializer()

x_vals_train = []
y_vals_train = []
x_vals_test = []
y_vals_test = []


x_loss_vals_train = []
y_loss_vals_train = []
x_loss_vals_test = []
y_loss_vals_test = []
#train model
test_data = {x :mnist.test.images,y_:mnist.test.labels, pkeep: 1.0}
with tf.Session() as sess:
    sess.run(init_op)
    for i in xrange(training_steps+1):
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, pkeep: 0.75})
        sess.run(train_step, {x: batch_xs, y_: batch_ys, lr: learning_rate, pkeep: 0.75})
         
        if i%collect_interval == 0:
            accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys,pkeep: 1.0})
             
            x_vals_train.append(i)
            y_vals_train.append(accuracy_val.mean()*100)
            x_loss_vals_train.append(i)
            y_loss_vals_train.append(loss_val.min())
            
            
            accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=test_data)
            x_vals_test.append(i)
            y_vals_test.append(accuracy_val.mean()*100) 
            
            x_loss_vals_test.append(i)
            y_loss_vals_test.append(loss_val.min()) 
             
            
        if i%print_iter == 0:
            print("Iteration :  {} , Loss : {:.5} - {:.5} Accuracy :  {:.5} - {:.5}".format(i,loss_val.min() , loss_val.max(), accuracy_val.min()*100, accuracy_val.max()*100))
               
       
    test_data = {x :mnist.test.images,y_:mnist.test.labels, pkeep:1.0}
    accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=test_data)
    print("Result on test data")
    print("Loss : {:.5} - {:.5} Accuracy :  {:.5} - {:.5}".format(loss_val.min() , loss_val.max(), accuracy_val.min()*100, accuracy_val.max()*100))
    print(dir(loss_val))
    save_path = saver.save(sess, "./output/model.ckpt")
       
    print ("Model saved in file: ", save_path)

end = time.time()



fig = plt.figure(num=None, figsize=(16, 9), dpi=360, facecolor='w', edgecolor='k')

fig.subplots_adjust(bottom=0.2)
#fig.suptitle("TensorFlow Accuracy graph for mnist.")

### start first plot
plt.subplot(2,1,1)
linecos1 = plt.plot(x_vals_train, y_vals_train, 'b-', label='Accuracy (Train data)')
linecos2 = plt.plot(x_vals_test, y_vals_test, 'g-', label='Accuracy (Test data)')
plt.xlabel("Training step")
plt.ylabel("Accuracy ( Percentage %)")
plt.ylim(0,100)
x_limit = max(max(max(x_vals_train), max(x_vals_test)), 105)
plt.xlim(0,x_limit)
interval = x_limit/20.0
plt.xticks(np.arange(0, x_limit, interval), rotation=20)
plt.yticks(np.arange(0, 105, 5.0)) 
plt.minorticks_on()
plt.legend()
plt.grid()


### start second plot
plt.subplot(2,1,2)
linecos1 = plt.plot(x_loss_vals_train, y_loss_vals_train, 'b-', label='Loss (Train data)')
linecos2 = plt.plot(x_loss_vals_test, y_loss_vals_test, 'g-', label='Loss (Test data)')
plt.xlabel("Training step")
plt.ylabel("Loss ")
x_limit = max(max(max(x_loss_vals_train), max(x_loss_vals_test)), 105)
plt.xlim(0,x_limit)
interval = x_limit/20.0
plt.xticks(np.arange(0, x_limit, interval), rotation=20)
plt.yticks(np.arange(0, max(y_loss_vals_train), max(y_loss_vals_train)/10.0))
plt.minorticks_on()
plt.legend() 
plt.grid()

### stop second plot

#print(x_vals)
#print(y_vals)
plt.figtext(0.10, 010, "Total time taken : %s seconds."%(int(end-start)), horizontalalignment='left') 
plt.figtext(0.10, 0.08, "Optimizer  : {}.".format(learning_fxn), horizontalalignment='left') 
plt.figtext(0.10, 0.06, "Accuracy on test data set  : {:.5} percent.".format(accuracy_val.mean()*100), horizontalalignment='left') 
plt.figtext(0.10, 0.04, "Total loss:  %s."%(loss_val.mean()),horizontalalignment='left') 
plt.figtext(0.10, 0.02, "Number of training steps :  %s, Learning rate :  %s."%(training_steps, learning_rate),horizontalalignment='left') 


stamp = "_".join(time.asctime().split())
file_name = "./output/tensor_accuracy_"+os.path.basename(__file__).replace(".py" , "_")+dataset_name+"_"+stamp+".png"

end = time.time()

print("graph saved in : ", file_name)
print("Time taken : ", int(end - start))
plt.savefig(file_name)
if update_test_result:
    update_results(dataset_type, activation_fxn,nums_of_layer,learning_fxn,learning_rate,batch_size,training_steps, accuracy_val.mean()*100, loss_val.mean(), int(end-start))
print("Done.")
#update_results(accuracy_val.mean()*100, loss_val.max())








###


   

