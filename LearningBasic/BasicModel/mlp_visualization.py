# -*- coding: utf-8 -*-

"""
Graph and Loss visualization using Tensorboard
Using MNIST database and Multilayer Perception
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#================================ Parameter ===================================

# Parameter
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = '/mlp_log'

# Network Parameters
n_hidden1 = 256
n_hidden2 = 256
n_input = 784
n_classes = 10

#================================ Define Model ================================
print("====================Building Model====================")
# TensorFlow Graph
x = tf.placeholder(tf.float32,[None,784],name="InputData")
y = tf.placeholder(tf.float32,[None,10],name="LabelData")

def mlp(x,weights,biases):
    layer1 = tf.add(tf.matmul(x,weights['w1']),biases['b1'])
    layer1 = tf.nn.relu(layer1)
    
    # Create a summary to visualize the first layer ReLU activation
    tf.histogram_summary("relu1",layer1)
    
    layer2 = tf.add(tf.matmul(layer1,weights['w2']),biases['b2'])
    layer2 = tf.nn.relu(layer2)
    
    tf.histogram_summary("relu2",layer2)
    
    output_layer = tf.add(tf.matmul(layer2,weights['w3']),biases['b3'])
    return output_layer
    
weights = {
    'w1':tf.Variable(tf.random_normal([n_input,n_hidden1]),name="W1"),
    'w2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2]),name="W2"),
    'w3':tf.Variable(tf.random_normal([n_hidden2,n_classes]),name="W3"),
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden1]),name="b1"),
    'b2':tf.Variable(tf.random_normal([n_hidden2]),name="b2"),
    'b3':tf.Variable(tf.random_normal([n_classes]),name="b3")
}

# Encapsulating all ops into scopes,making Tensorboard's Graph
with tf.name_scope("Model1"):
    pred = mlp(x,weights,biases)

with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))

with tf.name_scope("SGD"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Calculate every variable gradient    
    grads = tf.gradients(loss,tf.trainable_variables())
    grads = list(zip(grads,tf.trainable_variables()))
    # Update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
    
with tf.name_scope("Accuracy"):
    acc = tf.equal(tf.argmax(pred,1),tf.arg_max(y,1))
    acc = tf.reduce_mean(tf.cast(acc,tf.float32))

print("====================Initializating====================")
# Init
init = tf.initialize_all_variables()

# Create a summary to monitor cost,accuracy tensor
tf.scalar_summary("loss",loss)
tf.scalar_summary("accuracy",acc)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.histogram_summary(var.name,var)
    
# Create summaries to visualize all gradients
for grad,var in grads:
    tf.histogram_summary(var.name+'/gradient',grad)

# Mearge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()

#==================================== Launch Model ============================
print("====================Launching Model====================")
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path,
                                            graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            xs,ys = mnist.train.next_batch(batch_size)
            _,c,summary = sess.run([apply_grads,loss,merged_summary_op],
                                   feed_dict={x:xs,
                                              y:ys})
            summary_writer.add_summary(summary,epoch * total_batch + i)
            avg_cost += c/total_batch
        if (epoch+1)%display_step == 0:
            print("Epoch:{0},cost={1}".format(epoch,avg_cost))
    print("Optimization Done")
    
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    