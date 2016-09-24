# -*- coding: utf-8 -*-
"""
TensorFlow中的Word2Vec，使用Text8数据集
CBOW模型，使用上下文的词推测当前词
"""
import zipfile
import tensorflow as tf
import numpy as np
import random
import math
import collections

from matplotlib import pylab
from sklearn.manifold import TSNE

# global variable
data_index = 0 # 采样时从文本的第一个词开始采样

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
    
def build_dataset(vocabulary_size,words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

"""
data 整个文本文件经过build_dataset之后转换成的词在字典中的索引
batch_size 批量大小
num_skips 重复使用输入的次数(表示input用了产生label的次数限制)
    :num_skips = 2时，每个词右侧出现两次即由当前词可以推测出两个最可能的词，对应窗口左右各为1
    :num_skips = 4时，每个词右侧出现四次即由当前词可以推测出4个最可能的词，对应窗口左右各位2
skip_window CBOW模型左右上下文的词的个数(窗口大小)
"""
def generate_batch(data,batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    context_size = 2 * skip_window
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    batchs = np.ndarray(shape=(context_size,batch_size),dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1)%(len(data))
    
    for i in range(batch_size//num_skips):
        target = skip_window
        for j in range(num_skips):
            labels[i * num_skips + j,0] = buffer[target]
            met_target = False
            for k in range(context_size):
                if k == target:
                    met_target = True
                if met_target == True:
                    batchs[k,i * num_skips + j] = buffer[k+1]
                else:
                    batchs[k,i * num_skips + j] = buffer[k]
        buffer.append(data[data_index])
        data_index = (data_index + 1)%len(data)
    return batchs,labels


    
def train(filename="text8.zip",
          vocabulary_size=50000,
          embedding_size=128,# Dimension of the embedding vector
          batch_size=128,
          skip_window=1,# 上下文大小
          num_skips=2,# 每个词采样时作为中心词使用的次数)
          valid_size=16,# 随机选取的词的个数去评估模型
          valid_window=100,# 从当前词的前多少个选取测试样本
          num_sampled=64,# 负采样数
          num_steps=100001
          ):
    print('...... Reading data from zip file......')
    words = read_data(filename)
    print('Data Size = {0}'.format(len(words)))
    
    print('...... Transfer word data to word index,dictionary ......')
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary_size,words)
    
    print('...... Building model ......')
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):
        # Input data        
        train_dataset = tf.placeholder(tf.int32,shape=[2 * skip_window,batch_size])
        train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
        valid_dataset = tf.constant(valid_examples,shape=[2*skip_window,batch_size],dtype=tf.int32)
        
        # Varibles
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0)        
        )
        softmax_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size,embedding_size],
                                stddev=1.0/math.sqrt(embedding_size))        
        )
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
        # Model
        embed = tf.nn.embedding_lookup(embeddings,train_dataset)
        embed_sum = tf.reduce_mean(embed,0)
        
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(softmax_weights,softmax_biases,
                                       embed_sum,
                                       train_labels,num_sampled,vocabulary_size)        
        )
        
        optimizer = tf.train.AdadeltaOptimizer(1.0).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        # sum up vectors
        valid_embeddings_sum = tf.reduce_sum(valid_embeddings, 0)
        similarity = tf.matmul(valid_embeddings_sum, tf.transpose(normalized_embeddings))
    
    print('...... Training......')
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        average_loss = 0
        for step in range(num_steps):
            batch_data, batch_labels = generate_batch(data,
                batch_size, num_skips, skip_window)
            # print(batch_data.shape)
            # print(batch_labels.shape)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
        final_embeddings = normalized_embeddings.eval()
        save_obj("text8_cbow.pickle",final_embeddings)
    return final_embeddings,reverse_dictionary
        
        
def save_obj(pickle_file,obj):
    import cPickle as pickle
    import os
    try:
        f = open(pickle_file, 'wb')
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:{0}MB'.format(statinfo.st_size/(1024.*1024.)))
    
def tsne_and_plot(num_points,
                  final_embeddings,
                  labels):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])
    assert two_d_embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = two_d_embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()
    
if __name__ == "__main__":
    num_points = 400
    final_embeddings,reverse_dictionary = train()

    words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
    tsne_and_plot(num_points,final_embeddings,words)