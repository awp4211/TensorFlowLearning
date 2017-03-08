"""
Deep Neural Decision Forest
write with tensorflow 1.0
"""

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data


#parameter setting
DEPTH = 3               #DEPTH OF a tree
N_LEAF = 2**(DEPTH +1)  # number of leaf node
N_LABEL = 10            # number of classes
N_TREE = 5              # number of trees(ensemble)
N_BATCH = 128           # number of data points per mini-batch

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def init_prob_Weights(shape,minval=-5,maxval=5):
    return tf.Variable(tf.random_uniform(shape,minval=minval,maxval=maxval))

def model(X,w,w2,w3,w4_e,w_d_e,w_l_e,p_keep_conv,p_keep_hidden):
    """
    Create a forest and return the neural decision forest outputs:

    :param X:   input
    :param w:   conv1 filter
    :param w2:  conv2 filter
    :param w3:  conv3 filter
    :param w4_e:
    :param w_d_e:
    :param w_l_e:
    :param p_keep_conv:
    :param p_keep_hidden:
    :return:
        decision_p_e:decision node routing probability for all ensemble
            if we number all nodes in the tree sequentially from top to bottom,
            left to right ,decision_p contains
            [d(0),d(1),d(2)...d(2^n-2)]

            decision_p_e is the concatenation of all tree decision_p

        leaf_p_e:terminal node probability distributions for all ensemble.
            The indexing is the same as that of decision_p_e.
    """

    print('...... parameter shape ......')
    print('...... conv1 filter shape = {0}'.format(w.get_shape()))#(3,3,1,32)
    print('...... conv2 filter shape = {0}'.format(w2.get_shape()))#(3,3,32,64)
    print('...... conv3 filter shape = {0}'.format(w3.get_shape()))#(3,3,64,128)
    print('...... w4_e element shape = {0}'.format(w4_e[0].get_shape()))# (2048,625)
    print('...... w_d_e element shape = {0}'.format(w_d_e[0].get_shape()))# (625,16)
    print('...... w_l_e element shape = {0}'.format(w_l_e[0].get_shape()))#(16,10)

    assert(len(w4_e) == len(w_d_e))
    assert(len(w4_e) == len(w_l_e))

    print('...... initialization MNIST data shape = {0}'.format(X.get_shape()))
    conv1 = tf.nn.relu(tf.nn.conv2d(X,w,[1,1,1,1],'SAME'))
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1 = tf.nn.dropout(conv1,p_keep_conv)

    print('...... conv1 shape = {0}'.format(conv1.get_shape()))#(batch,14,14,32)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1,w2,[1,1,1,1],'SAME'))
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2 = tf.nn.dropout(conv2,p_keep_conv)

    print('...... conv2 shape = {0}'.format(conv2.get_shape()))#(batch,7,7,64)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2,w3,strides=[1,1,1,1],padding='SAME'))
    conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    print('...... conv3 shape = {0}'.format(conv3.get_shape()))#(batch,4,4,128)

    conv3 = tf.reshape(conv3,[-1,w4_e[0].get_shape().as_list()[0]])
    conv3 = tf.nn.dropout(conv3,p_keep_conv)

    print('...... conv3 flatten shape = {0}'.format(conv3.get_shape()))#(128,2048=4*4*128)

    decision_p_e = []
    leaf_p_e = []
    for w4,w_d,w_l in zip(w4_e,w_d_e,w_l_e):
        net = tf.nn.relu(tf.matmul(conv3,w4))
        net = tf.nn.dropout(net,p_keep_hidden)

        print('...... decision_p before sigmoid,shape = {0}'.format(net.get_shape()))
        decision_p = tf.nn.sigmoid(tf.matmul(net,w_d))
        print('...... decision_p(decision node rounting probability) shape = {0}'.format(decision_p.get_shape()))

        leaf_p = tf.nn.softmax(w_l)
        print('...... leaf_p(leaf probability distribute shape = {0}'.format(leaf_p.get_shape()))

        decision_p_e.append(decision_p)
        leaf_p_e.append(leaf_p)

    return decision_p_e,leaf_p_e


def train():

    print('...... loading dataset .......')
    mnist = input_data.read_data_sets("MNIST/",one_hot=True)
    trX,trY = mnist.train.images,mnist.train.labels
    teX,teY = mnist.test.images,mnist.test.labels
    trX = trX.reshape(-1,28,28,1)
    teX = teX.reshape(-1,28,28,1)

    ######################################
    #input placeholder
    X = tf.placeholder(tf.float32,[N_BATCH,28,28,1])
    Y = tf.placeholder(tf.float32,[N_BATCH,N_LABEL])

    w = init_weights([3,3,1,32])
    w2 = init_weights([3,3,32,64])
    w3 = init_weights([3,3,64,128])

    w4_ensemble = []
    w_d_ensemble = []
    w_l_ensemble = []
    for i in range(N_TREE):
        w4_ensemble.append(init_weights([128*4*4,625]))
        w_d_ensemble.append(init_prob_Weights([625,N_LEAF],-1,1))
        w_l_ensemble.append(init_prob_Weights([N_LEAF,N_LABEL],-2,2))

    p_keep_conv = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)

    #######################################
    #define a fully differentiable deep-NDF
    decision_p_e,leaf_p_e = model(X,w,w2,w3,
                                          w4_ensemble,
                                          w_d_ensemble,
                                          w_l_ensemble,
                                          p_keep_conv,p_keep_hidden)

    flat_decision_p_e = []

    # iterate over each tree
    for decision_p in decision_p_e:
        # compute the complement of d,which is 1-d where d is the sigmoid of fully connected output
        # tf.ones_like():all elements set to 1
        decision_p_comp = tf.subtract(tf.ones_like(decision_p),decision_p)
        print('...... decision_p_comp,shape = {0}'.format(decision_p_comp.get_shape()))

        # concatenate both d,1-d
        # tf.pack ==> tf.stack
        decision_p_pack = tf.stack([decision_p,decision_p_comp])
        print('...... decision_p_pack,shape = {0}'.format(decision_p_pack.get_shape()))#(2,128,16)

        # flatten/vectorize the decision probabilities for efficient indexing
        # vectorize (decision_p,decision_p_comp) to a 1 dimension vector
        flat_decision_p = tf.reshape(decision_p_pack,[-1])
        flat_decision_p_e.append(flat_decision_p)

    print('...... flat decision_p_e element,shape = {0}'.format(flat_decision_p_e[0].get_shape()))
    # index of each data instance in a mini-batch
    # tf.tile(input,multiples):output tensor i'th dimension has input.dim(i)*multiples[i] elements.
    #   and the values of input are replicated multiples[i] times along the i'th dimension
    # tf.range(start,limit,delta):generate a sequence begins at start and extends by increments of delta.
    # tf.expand_dims():given a tensor input, this operation inserts a dimension of 1 at the dimension index axis of input's shape
    batch_0_indices = tf.tile(tf.expand_dims(tf.range(0,N_BATCH * N_LEAF,N_LEAF),1),[1,N_LEAF])
    # tf.range() ===> a list
    # tf.expand_dims(tf.range()) ===> [?,1] array
    # tf.tile(tf.expand_dims(tf.range())) ===> [?,N_LEAF] array,each element is the first index
    print('...... batch_0_indices,shape = {0}'.format(batch_0_indices.get_shape()))

    ####################################
    # the routing probability computation
    # firstly create a routing probability matrix. \mu
    # initialize matrix using the root node d,1-d.To efficiently implement this routing,
    # we will create a giant vector(matrix) that contains all d and 1-d from all
    # decision nodes.The matrix version of that is decision_p_pack and vectorized
    # version is flat_decision_p.

    # The suffix  `_e` indicates an ensemble. i.e. concatenation of all responsens from trees
    # For DEPTH =2 tree,the routing probability for each leaf node can be easily compute
    # by multiplying the following vectors elementwise.(DEPTH=2,N_LEAF=8)
    # mu =      [d_0,  d_0,   d_0,   d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
    # mu = mu * [d_1,  d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
    # mu = mu * [d_3,1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]

    # Tree indexing
    #      0
    #   1     2
    # 3  4   5  6
    #####################################
    in_repeat = N_LEAF /2
    out_repeat = N_BATCH

    # Let N_BATCH * N_LEAF be N_D. flat_decision_p[N_D] will return 1-d of the
    # first root node in the first tree
    batch_complement_indices = \
        np.array([[0]* in_repeat,[N_BATCH * N_LEAF] * in_repeat]
                 * out_repeat).reshape(N_BATCH,N_LEAF)

    # first define the routing probabilities d for root nodes
    mu_e = []

    # iterate over each tree
    for i,flat_decision_p in enumerate(flat_decision_p_e):
        mu = tf.gather(flat_decision_p,
                               tf.add(batch_0_indices,batch_complement_indices))
        mu_e.append(mu)

    # from the second layer to the last layer,we make the decision nodes
    for d in xrange(1,DEPTH+1):
        indices = tf.range(2**d,2**(d+1))-1
        tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices,1),
                                                  [1,2**(DEPTH - d + 1)]),[1,-1])
        batch_indices = tf.add(batch_0_indices,tf.tile(tile_indices,[N_BATCH,1]))

        in_repeat = in_repeat / 2
        out_repeat = out_repeat * 2

        # again define the indices that picks d and 1-d for the node
        batch_complement_indices = \
            np.array([[0]*in_repeat,[N_BATCH * N_LEAF] * in_repeat]
                             * out_repeat).reshape(N_BATCH,N_LEAF)
        mu_e_update = []
        for mu,flat_decision_p in zip(mu_e,flat_decision_p_e):
            mu = tf.multiply(mu,tf.gather(flat_decision_p,
                                             tf.add(batch_indices,batch_complement_indices)))
            mu_e_update.append(mu)

        mu_e = mu_e_update

    ######################################
    # define p(y|x)
    py_x_e = []
    for mu,leaf_p in zip(mu_e,leaf_p_e):
        py_x_tree = tf.reduce_mean(
            tf.multiply(
                    tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABEL]),
                    tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1])), 1)
        py_x_e.append(py_x_tree)

    py_x_e = tf.stack(py_x_e)
    py_x = tf.reduce_mean(py_x_e,0)

    #######################################
    # define cost and optimization method

    # cross entropy
    cost = tf.reduce_mean(-tf.multiply(tf.log(py_x),Y))

    # cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(py_x,Y))
    train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
    predict = tf.argmax(py_x,1)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in range(100):
        # one epoch
        for start,end in zip(range(0,len(trX),N_BATCH),range(N_BATCH,len(trX),N_BATCH)):
            sess.run(train_step,feed_dict={X:trX[start:end],
                                           Y:trY[start:end],
                                           p_keep_conv:0.8,
                                           p_keep_hidden:0.5})
        results = []
        for start,end in zip(range(0,len(teX),N_BATCH),range(N_BATCH,len(teX),N_BATCH)):
            results.extend(np.argmax(teY[start:end],axis=1)
                           == sess.run(predict,feed_dict={X:teX[start:end],
                                                          p_keep_conv:1.0,
                                                          p_keep_hidden:1.0}))
        print 'Epoch:%d,Test Accuracy:%f'%(i+1,sum(results)/len(results))


if __name__ == '__main__':
    train()

