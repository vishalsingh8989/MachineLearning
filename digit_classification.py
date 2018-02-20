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
__status__ = "working"


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from testresults import update_results
from sanity import Sanity




activation_fxn = "softmax"
nums_of_layer = 1
learning_fxn = "GradientDescentOptimizer"
dataset_type = "MNIST"
dataset_name = "MNIST_data"

learning_rate = 0.003
batch_size = 100
pic_number = 10   # neurals count
pic_size = 28
training_steps = 2000
print_iter = 1


update_test_result = True
plotgraph = True
 
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
cross_entropy =-tf.reduce_sum(y_*tf.log(y)) 


train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
   
   
# correct answers
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
   
init_op = tf.initialize_all_variables()

saver = tf.train.Saver()
   
   
x_vals_train = []
y_vals_train = []
x_vals_test = []
y_vals_test = []


x_loss_vals_train = []
y_loss_vals_train = []
x_loss_vals_test = []
y_loss_vals_test = []
 
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
            x_loss_vals_train.append(i)
            y_loss_vals_train.append(loss_val.min())
            
            
            accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=test_data)
            x_vals_test.append(i)
            y_vals_test.append(accuracy_val.mean()*100) 
            
            x_loss_vals_test.append(i)
            y_loss_vals_test.append(loss_val.min()) 
             
           
         
            print("Iteration :  {} , Loss : {:.5} - {:.5} Accuracy :  {:.5} - {:.5}".format(i,loss_val.min() , loss_val.max(), accuracy_val.min()*100, accuracy_val.max()*100))
               
       
    test_data = {x :mnist.test.images,y_:mnist.test.labels}
    accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=test_data)
    print("Result on test data")
    print("Loss : {:.5} - {:.5} Accuracy :  {:.5} - {:.5}".format(loss_val.min() , loss_val.max(), accuracy_val.min()*100, accuracy_val.max()*100))
    
    save_path = saver.save(sess, "./output/model.ckpt")
       
    print ("Model saved in file: ", save_path)


end = time.time()

if not plotgraph:
    exit()

fig = plt.figure(num=None, figsize=(16, 9), dpi=360, facecolor='w', edgecolor='k')

fig.subplots_adjust(bottom=0.2)
#fig.suptitle("TensorFlow Accuracy graph for mnist.")

### start first plot
plt.subplot(2,1,1)
linecos1 = plt.plot(x_vals_train, y_vals_train, 'r-', label='Accuracy (Train data)')
linecos2 = plt.plot(x_vals_test, y_vals_test, 'b-', label='Accuracy (Test data)')
plt.xlabel("Training step")
plt.ylabel("Accuracy ( Percentage %)")
plt.ylim(0,100)
x_limit = max(max(max(x_vals_train), max(x_vals_test)), 105)
plt.xlim(0,x_limit)
interval = x_limit/20.0
plt.xticks(np.arange(0, x_limit, interval), rotation=20)
plt.yticks(np.arange(0, 105, 5.0)) 
#plt.minorticks_on()
plt.legend()
plt.grid()


### start second plot
plt.subplot(2,1,2)
linecos1 = plt.plot(x_loss_vals_train, y_loss_vals_train, 'r-', label='Loss (Train data)')
linecos2 = plt.plot(x_loss_vals_test, y_loss_vals_test, 'b-', label='Loss (Test data)')
plt.xlabel("Training step")
plt.ylabel("Loss ")
x_limit = max(max(max(x_loss_vals_train), max(x_loss_vals_test)), 105)
plt.xlim(0,x_limit)
interval = x_limit/20.0
plt.xticks(np.arange(0, x_limit, interval), rotation=20)
plt.yticks(np.arange(0, max(y_loss_vals_train), max(y_loss_vals_train)/10.0))
#plt.minorticks_on()
plt.legend() 
plt.grid()

### stop second plot

#print(x_vals)
#print(y_vals)
plt.figtext(0.10, 0.10, "Total time taken : %s seconds."%(int(end-start)), horizontalalignment='left') 
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
print("Done")
if update_test_result:
    update_results(dataset_type, activation_fxn,nums_of_layer,learning_fxn,learning_rate,batch_size,training_steps, accuracy_val.mean()*100, loss_val.mean())

