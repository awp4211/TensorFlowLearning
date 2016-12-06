# -*- coding: utf-8 -*-

"""
Convolutional AutoEncoder(CAE)
"""
import tensorflow as tf
import numpy as np
import math

def lrelu(x,leak=0.2,name='lrelu'):
    """
    Leaky rectifier
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
        
def corrupt(x):
    return tf.mul(x,tf.cast(tf.random_uniform(shape=tf.shape(x),
                                              minval=0,
                                              maxval=2,
                                              dtype=tf.int32),tf.float32))
                                              
def autoencoder(input_shape=[None,784],
                n_filters=[1,10,10,10],
                filter_sizes=[3,3,3,3],
                corruption=False):
    """
    Building a deep denoising autoencoder w-tied weights
    Returns:
        x:Tensor:Input placeholder to the network
        z:Tensor:Inner-most latent representation
        y:Tensor:Output reconstruction fo the input
        cost:Tensor:Overall cost to use for training
    """
    x = tf.placeholder(tf.float32,input_shape,name='x')
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsopported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(x,[-1,x_dim,x_dim,n_filters[0]])

    current_input = x_tensor
    if(corruption):
        current_input = corrupt(current_input)
        
    encoder = []
    shapes = []
    
    print('...... Building the encoder ......')
    for layer_i,n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([filter_sizes[layer_i],filter_sizes[layer_i],
                               n_input,n_output],
                               -1.0 / math.sqrt(n_input),
                                1.0 / math.sqrt(n_input))        
        )
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(tf.add(tf.nn.conv2d(current_input,W,
                                           strides=[1,2,2,1],
                                           padding='SAME'),b
                                           ))
        print('Encoder layer_{0},shape={1}'.format(layer_i,output.get_shape()))
        current_input = output
        
        
    z = current_input
    
    encoder.reverse()
    shapes.reverse()
    
    print('...... Building the decoder ......')
    for layer_i,shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input,W,
                tf.pack([tf.shape(x)[0],shape[1],shape[2],shape[3]]),
                strides=[1,2,2,1],padding='SAME'),b))
        print('Decoder layer_{0},shape={1}'.format(layer_i,output.get_shape()))
        current_input = output
        
        
    y = current_input
    cost = tf.reduce_sum(tf.square( y - x_tensor))
    
    return {'x':x,'y':y,'z':z,'cost':cost}
    
def mlp(x,hidden_size=[500,200,10]):
    """
    Multilayer perceptron
    x:Tensor:[batch_size,width*height*channels]
    
    """
    current_input = x
    for layer_i,n_output in enumerate(hidden_size):
        n_input = int(current_input.get_shape()[1])
        val = math.sqrt(float(n_input+n_output))
        weight = tf.Variable(tf.random_normal([n_input,n_output],
                                              -1.0 / val,
                                               1.0 / val))
        bias = tf.Variable(tf.constant(0.1,shape=[n_output,]))
        output = tf.matmul(current_input,weight)
        output = tf.add(output,bias)
        current_input = output
        
    return current_input
    
def test_mnist():
    """
    Test the convolutional autoencoder using MNIST
    """
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    mean_img = np.mean(mnist.train.images,axis=0)
    ae = autoencoder()
    
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    
    print('...... Initialize all variables ......')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    
    print('...... Start to training ......')    
    batch_size = 100
    n_epochs = 3
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs,_ = mnist.train.next_batch(batch_size)
            #均值图像
            train = np.asarray([img - mean_img for img in batch_xs])
            sess.run(optimizer,feed_dict={ae['x']:train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))
    
    """
    SAVE Model
    """
    _file = 'Yawn_dataset/cae.npy'    
    cae_inner = []
    batch_size = 100
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs,_ = mnist.train.next_batch(batch_size)
        train = np.asarray([img - mean_img for img in batch_xs])
        cae_z = sess.run(ae['z'],feed_dict={ae['x']:train})
        print(cae_z.shape)
        for vec in cae_z:
            cae_inner.append(vec)
    np.save(_file,cae_inner)
    
    sess.close()
    """
    SAVE Model end
    """
   
    
    """
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()
    """

def classify_mnist(training_epoch=10,batch_size=100):
    """
    MLP
    """
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    
    print('...... Loading dataset ......')    
    
    _file = 'Yawn_dataset/cae.npy'
    test_set_x = np.load(_file)
    test_set_x = test_set_x.reshape([-1,test_set_x.shape[1]*
                                        test_set_x.shape[2]*
                                        test_set_x.shape[3]])
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
    n_classes = 10
    n_input = test_set_x.shape[1]
    x = tf.placeholder(tf.float32,[None,n_input])
    y = tf.placeholder(tf.float32,[None,n_classes])
    
    
    print('...... Building model ......')
    pred = mlp(x,hidden_size=[500,200,10])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    init  = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        print('...... Training ......')
        for epoch in range(training_epoch):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_xs = test_set_x[i*batch_size:(i+1)*batch_size]
                _,batch_ys = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,
                                                       y:batch_ys})
                avg_cost += c/total_batch
            print("epoch_{0},cost={1}".format(epoch,avg_cost))
        
        print('...... Optimize Done ......')
        correct_prediction = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print("Accuracy:",accuracy.eval({x:test_set_x,y:mnist.train.labels}))
            
        
    
        
if __name__ == '__main__':
    #test_mnist()
    classify_mnist()