#Intro to TensorFlow Core 
#linear regression - find best linear function to fit data 
#use gradient descent as optimizer
#sum of squares of distances as loss

import numpy as np
import tensorflow as tf

#Model parameters
W = tf.Variable([0], dtype=tf.float32)
b = tf.Variable([-.5], dtype=tf.float32)
#Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
#loss
loss = tf.reduce_sum(tf.square(linear_model - y)) #sum of squares
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) #.01 = step size
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [-.5,0,.5,1] #perfect W and b to get 0 loss is W=0.5, b=1
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

#used code from https://www.tensorflow.org/get_started/get_started
