# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

n_inputs = 28 # MNIST data input(img shape:28*28)
n_steps = 28 # Time steps
n_hidden_units = 128 # neurons in hidden layer
n_classes = 10 # MNIST classes(0-9 digits)

# tf Graph Input
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

# Define weights
weights = {
    #(28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    #(128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
biases = {
    #(128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    #(10,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}


def RNN(X,weights,biases):
    # hidden layer for input to cell
    # X(128 batches,28 steps,28 inputs)
    # X ==> (128 * 28,28 inputs)
    X = tf.reshape(X,[-1,n_inputs])
    

    # into hidden    
    # X_in ==> (128batch * 28 steps,128 hidden)
    X_in = tf.matmul(X,weights['in']) + biases['in']
    # X_in ==> (128batch,28steps,128hidden)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])
        
    # cell
    # forget_bias = 1.0表示起始时间所有信息通过
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,
                                             forget_bias=1.0)
    # lstm cell is divided into two parts(c_state,h_state)
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    # time_major=True表示X_in主要维度,即128batch
    # time_major=False表示X_in次要维度,即28steps（数据的n_steps在第二维，使用False）
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,
                                       X_in,
                                       initial_state=_init_state,
                                       time_major=False) 
    #==================================DUBUG===================================
    # print('!! 1',type(outputs))#tensorflow.python.framework.ops.Tensor
    # print('!! 2',type(states))#tensorflow.python.framework.ops.Tensor
                                       
    #[None,28,128](None=batch,28 steps,128 hidden_outputs)
    #print('!! 1',outputs.get_shape())
    #[None,256]
    #print('!! 2',states.get_shape())
    #==================================DUBUG===================================

                    
    # hidden layer for output as the final results
    # unpack to list[(batch,outputs)* steps]
    # transpose:[batch,28steps,128hidden] ==> [28steps,batch,hidden]
    outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))   
    #==================================DUBUG===================================   
    #print(type(outputs))#list    
    #print(len(outputs))#n_steps
    #print(type(outputs[0]))#tensorflow.python.framework.ops.Tensor
    #print(outputs[0].get_shape())#[batch,hidden]
    #经过上述转换，output变成了[(batch,outputs)* steps]的list，outputs[-1]表示最后一个
    #step运行之后LSTM单元输出的结果,之后使用SOFTMAX回归即可得到相应的分类结果数据
    #==================================DUBUG===================================                         
    results = tf.matmul(outputs[-1],weights['out']) + biases['out']
    return results

pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0 
    best_acc = 0.    
    
    while step*batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys
        })
        if step % display_step == 0:
            acc,loss = sess.run([accuracy,cost],feed_dict={
                x: batch_xs, 
                y: batch_ys,
            })
            if acc > best_acc:
                best_acc = acc
            print("Iter {0},Minibatch Loss = {1},Training Accuracy = {2}".format(
                step*batch_size,loss,acc))
        step+=1
    print("Optimization Done!")
    print("Best Accuracy:{0}".format(best_acc))
    
    test_len = 100
    test_data_size = 10000
    correct_count = 0.
    
    for index in range(test_data_size/test_len):
        test_data = mnist.test.images[test_len*index:test_len*(index+1)].reshape((-1,n_steps,n_inputs))
        test_label = mnist.test.labels[test_len*index:test_len*(index+1)]
        acc_t = sess.run(accuracy,feed_dict={
                x:test_data,
                y:test_label,
            })
        correct_count = correct_count + acc_t * test_len
        
    print('Testing Correct count = {0}'.format(correct_count))
    print("Testing Accuracy:{0}".format(correct_count/test_data_size))