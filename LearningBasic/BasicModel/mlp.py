# -*- coding: utf-8 -*-

"""
A Multilayer Perceptron implementation example using TensorFlow Library
This example is using the MNIST database
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters

n_hidden1 = 256 # 1st layer number of features
n_hidden2 = 256 # 2nd layer number of features
n_input = 784   # MNIST data input(img shape 28*28)
n_classes = 10  # MNIST total classes 

# TensorFlow Graph input
x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

# Create Model
def multilayer_perceptron(x,weights,biases):
    # Hidden layer with RELU activation
    layer1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer1 = tf.nn.relu(layer1)
    # Hidden layer with RELU activation
    layer2 = tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
    layer2 = tf.nn.relu(layer2)
    # Output layer with linear activiation
    out_layer = tf.matmul(layer2,weights['out'])+biases['out']
    return out_layer
    
# Store layers weight and bias
weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden1])),
    'h2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
    'out':tf.Variable(tf.random_normal([n_hidden2,n_classes]))
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden1])),
    'b2':tf.Variable(tf.random_normal([n_hidden2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}
# Construct model
pred = multilayer_perceptron(x,weights,biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the varibales
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # Training Cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,
                                                       y:batch_y})
            avg_cost += c/total_batch
        if epoch % display_step == 0:
             print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("Optimization Done")
    
    # Test model
    # tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的
    # 索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)
    correct_prediction = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))

    # print(type(correct_prediction)) <class 'tensorflow.python.framework.ops.Tensor'>

    # Calculate accuracy 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    
    # print(type(accuracy)) <class 'tensorflow.python.framework.ops.Tensor'>
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))