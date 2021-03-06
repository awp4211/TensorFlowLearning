# -*- coding: utf-8 -*-
"""
Using RNN models to predict continues Data like sine and cosine
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_start = 0
time_steps = 20
batch_size = 50
input_size = 1
output_size = 1
cell_size = 10
lr = 0.006

def get_batch():
    global batch_start,time_steps
    # xs shape(50batch,20steps)
    xs = np.arange(batch_start,batch_start + time_steps * batch_size)
    xs = xs.reshape((batch_size,time_steps)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    batch_start += time_steps
    
    #================================DISPLAY=================================== 
    #plt.ion()
    #plt.plot(xs[0,:],res[0,:],'r',xs[0,:],seq[0,:],'b--')
    #plt.show()
    #================================DISPLAY===================================    
    
    # returned seq res and xs:shape(batch,time_step,input)    
    # seq[:,:,np.newaxis] shape ==> (50,20,1)
    # res[:,:,np.newaxis] shape ==> (50,20,1)
    # xs shape ==> (50,20)
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]
    
class LSTMRNN(object):
    def __init__(self,
                 n_steps,# 序列长度
                 input_size,# 输入数据的维度
                 output_size,
                 cell_size,
                 batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32,[None,n_steps,input_size],name='xs')
            self.ys = tf.placeholder(tf.float32,[None,n_steps,output_size],name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)
    
    # 向LSTM中传入3D数据
    def add_input_layer(self,):
        # [batch,n_steps,input_size] ==> [batch * n_step,input_size]
        l_in_x = tf.reshape(self.xs,[-1,self.input_size],name='to_2D')
        # Ws [input_size,cell_size]
        Ws_in = self._weight_variable(shape=[self.input_size,self.cell_size],
                                      name='Ws_in')
        # bs [cell_size,]
        bs_in = self._bias_variable([self.cell_size,])
        with tf.name_scope('Wx_plus_b'):
            # [batch*n_steps,input_size]*[input_size,cell_size]
            #   ==>[batch*n_steps,cell_size]
            l_in_y = tf.matmul(l_in_x,Ws_in) + bs_in
        # reshape l_in_y [batch*n_steps,cell_size] ==> [batch,n_steps,cell_size]
        self.l_in_y = tf.reshape(l_in_y,[self.batch_size,self.n_steps,self.cell_size],name='to_3D')

    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size,forget_bias=1.0)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size,dtype=tf.float32)
        self.cell_outputs,self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell,
            self.l_in_y,
            initial_state=self.cell_init_state,
            time_major=False
        )
        #===================================DEBUG==============================
        print('!!!! ',self.cell_outputs.get_shape())
        print('!!!! ',self.cell_final_state.get_shape())
        #===================================DEBUG==============================
    
    def add_output_layer(self):
        # shape=[batch*n_steps,cell_size]
        l_out_x = tf.reshape(self.cell_outputs,[-1,self.cell_size],name='to_2D')
        Ws_out = self._weight_variable([self.cell_size,self.output_size])
        bs_out = self._bias_variable([self.output_size,])
        # shape=[batch*n_steps,outpt_size]
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x,Ws_out) + bs_out
    
    def compute_cost(self):
        losses = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.scalar_summary('cost', self.cost)
    
    
    def ms_error(self,y_pre,y_target):
        return tf.square(tf.sub(y_pre, y_target))
    
    def _weight_variable(self,shape,name='weights'):
        initializer = tf.random_normal_initializer(mean=0.,stddev=1.)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)
    
    def _bias_variable(self,shape,name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name,shape=shape,initializer=initializer)

if __name__ == '__main__':
    model = LSTMRNN(time_steps,input_size,output_size,cell_size,batch_size)
    sess = tf.Session()
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs",sess.graph)
    sess.run(tf.initialize_all_variables())
    
    plt.ion()
    plt.show()
    
    for i in range(200):
        seq,res,xs = get_batch()
        if i == 0:
            feed_dict = {
                model.xs : seq,
                model.ys : res,
            }
        else:
            feed_dict = {
                model.xs : seq,
                model.ys : res,
                model.cell_init_state:state
            }
        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        
        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)