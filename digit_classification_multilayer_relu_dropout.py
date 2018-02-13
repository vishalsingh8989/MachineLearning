from __future__ import absolute_import
from __future__ import print_function

"""handwriten_digits.py: Basic nueral networks for identifying handwriiten digits using tensorFlow
Activation: sigmoid and softmax
Accuracy acheived  :  97.24 %. training_steps = 100000

Test result : 
 ____________________________________________________________________________________________________________________________________________________
|     Activation function             |  Number of layer   | Learning fxn |  Learning_rate  |  Batch_size |  Training_steps | Accuracy |   Total loss | 
|  sigmoid and softmax               |  4                 | GDO          |  0.0005         |   100       |    100000       |  96.57   |    1559.6    | 
|  sigmoid and softmax               |  4                 | GDO          |  0.0002         |   100       |    200000       |          |


"""


__author__ = "Vishal Jasrotia"
__date__   = "Feb, 06, 2018"
__organization__ = "Stony Brook Univerity, NY, USA"
__email__ = "jasrotia.vishal@stonybrook.edu"
__version__ = "0.1"


import os
import csv
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import tensorflow as tf

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#locals
from testresults import update_results

activation_fxn = "relu and softmax with decaying learning rate"
nums_of_layer = 4
learning_fxn = "GradientDescentOptimizer"
dataset_type = "MNIST"
dataset_name = "MNIST_data"
 


max_learning_rate = 0.005
min_learning_rate = 0.0001
decay_speed = 2000.0
learning_rate = 0.0005
batch_size = 100
pic_number = 10   # neurals count
pic_size = 28
training_steps = 20000
print_iter = 100
collect_interval = 10

update_test_result = True  #True if you want to updat result in ./result/test_result.csv


#moved to Testdata
# def update_results(dataset_type, activation_fxn,nums_of_layer,learning_fxn,learning_rate,batch_size,training_steps, accuracy, total_loss):
#     result = dataset_type + ", " + activation_fxn + ", "  + str(nums_of_layer) + " ," + learning_fxn + ", " + str(learning_rate) + ", " + str(batch_size) + ", " + str(training_steps) + ", " + str(accuracy)[:5] + " , " + str(total_loss) + "," + str((total_loss/600000.0)*100)[:5]     
#     with open('./results/test_results.csv', 'a') as csvfile:
#         #datawriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)  
#         #datawriter.write(result + "\n")
#         csvfile.write(result + "\n")
#         csvfile.close()
            

layer_1 = 200
layer_2= 100
layer_3 = 60
layer_4 = 30


#load data
mnist = input_data.read_data_sets(dataset_name+"/", one_hot=True)
  


# learning rate placeholder
decay_learning_rate = tf.placeholder(tf.float32)

#placeholder for shootout
pkeep = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([pic_size*pic_size, layer_1], stddev = 0.1))
B1 = tf.Variable(tf.zeros([layer_1]))
 
 
W2 = tf.Variable(tf.truncated_normal([layer_1, layer_2], stddev = 0.1))
B2 = tf.Variable(tf.zeros([layer_2]))


W3 = tf.Variable(tf.truncated_normal([layer_2, layer_3], stddev = 0.1))
B3 = tf.Variable(tf.zeros([layer_3]))


W4 = tf.Variable(tf.truncated_normal([layer_3, layer_4], stddev = 0.1))
B4 = tf.Variable(tf.zeros([layer_4]))

W5 = tf.Variable(tf.truncated_normal([layer_4, pic_number], stddev = 0.1))
B5 = tf.Variable(tf.zeros([pic_number]))

start  =   int(time.time())
 
#mnist data from handwriten digits


   

#model with diff weight and biases
x = tf.placeholder(tf.float32, [None, pic_size*pic_size])

#Y0 = tf.sigmoid(tf.matmul(x,W0) + B0)

Y1 = tf.nn.relu(tf.matmul(x,W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d,W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d,W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)


Ylogits = tf.matmul(Y4d, W5) + B5
y = tf.nn.softmax(Ylogits)


#y = tf.nn.softmax(tf.matmul(Y4,W5) + B5)

#W = tf.Variable(tf.zeros([pic_size*pic_size, pic_number]))
#b = tf.Variable(tf.zeros([pic_number]))

#y = tf.nn.softmax(tf.matmul(x, W) + b)

#loss function and optimizer
y_ = tf.placeholder(tf.float32, [None, pic_number])


##cross_entropy =-tf.reduce_sum(y_*tf.log(y)) # can be mean also



cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
   
#optimizer = tf.train.AdamOptimizer(learning_rate=decay_learning_rate)

#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)



optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)
   
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
test_data = {x :mnist.test.images,y_:mnist.test.labels, pkeep: 1.0}
with tf.Session() as sess:
    sess.run(init_op)
    for i in xrange(training_steps+1):
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, pkeep: 0.75})
        
         
        if i%collect_interval == 0:
            accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys,pkeep: 1.0})
             
            x_vals_train.append(i)
            y_vals_train.append(accuracy_val.mean()*100)
            
            
            accuracy_val,loss_val = sess.run([accuracy,cross_entropy], feed_dict=test_data)
            x_vals_test.append(i)
            y_vals_test.append(accuracy_val.mean()*100) 
             
            
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
fig.suptitle("TensorFlow Accuracy graph for mnist.")


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

#print(x_vals)
#print(y_vals)
#plt.figtext(0.20, 0.08, "TensorFlow Accuracy of batch of 100 set", horizontalalignment='right') 
plt.figtext(0.28, 0.08, "Entropy Optimizer  : {}.".format(learning_fxn), horizontalalignment='right') 
plt.figtext(0.24, 0.06, "Accuracy on test data set  : {:.3} percent.".format(accuracy_val.mean()*100), horizontalalignment='right') 
plt.figtext(0.16, 0.04, "Total loss:  %s"%(loss_val.mean()),horizontalalignment='right') 
plt.figtext(0.36, 0.02, "Number of training steps :  %s , Learning rate :  %s."%(training_steps, learning_rate),horizontalalignment='right') 


plt.minorticks_on()
plt.legend()

plt.grid()
#plt.show()
stamp = "_".join(time.asctime().split())
file_name = "./output/tensor_accuracy_"+os.path.basename(__file__).replace(".py" , "_")+dataset_name+"_"+stamp+".png"

print("graph saved in : ", file_name)
plt.savefig(file_name)
if update_test_result:
    update_results(dataset_type, activation_fxn,nums_of_layer,learning_fxn,learning_rate,batch_size,training_steps, accuracy_val.mean()*100, loss_val.mean(), int(end-start))
  
#update_results(accuracy_val.mean()*100, loss_val.max())



