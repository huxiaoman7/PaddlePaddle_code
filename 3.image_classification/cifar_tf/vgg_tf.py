# -*- coding: utf-8 -*-
"""
Created by huxiaoman 2017.12.12
vgg_tf.py:训练tensorflow版的vgg16网络，对cifar-10shuju进行分类
"""
from datetime import datetime
import math
import time
import tensorflow as tf
import cifar10

batch_size = 16
num_batches = 100

# 定义函数对卷积层进行初始化
# input_op : 输入数据 
# name : 该卷积层的名字，用tf.name_scope()来命名
# kh,kw : 分别是卷积核的高和宽
# n_out : 输出通道数
# dh,dw : 步长的高和宽
# p ： 是参数列表，存储VGG所用到的参数
# 采用xavier方法对卷积核权值进行初始化
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value # 获得输入图像的通道数
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
            shape = [kh, kw, n_in, n_out], dtype = tf.float32,
            initializer = tf.contrib.layers.xavier_initializer_conv2d())
        #  卷积层计算
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding = 'SAME')
        bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)
        biases = tf.Variable(bias_init_val, trainable = True, name = 'b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name = scope)
        p += [kernel, biases]
        return activation

# 定义函数对全连接层进行初始化
# input_op : 输入数据
# name : 该全连接层的名字
# n_out : 输出的通道数
# p : 参数列表 
# 初始化方法用 xavier方法
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
            shape = [n_in, n_out], dtype = tf.float32,
            initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape = [n_out],
            dtype = tf.float32), name = 'b')
        activation = tf.nn.relu_layer(input_op, kernel,  #  ???????????????
            biases, name = scope)
        p += [kernel, biases]
        return activation 

# 定义函数 创建 maxpool层
# input_op : 输入数据 
# name : 该卷积层的名字，用tf.name_scope()来命名
# kh,kw : 分别是卷积核的高和宽
# dh,dw : 步长的高和宽
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize = [1,kh,kw,1],
        strides = [1, dh, dw, 1], padding = 'SAME', name = name)

#---------------创建 VGG-16------------------

def inference_op(input_op, keep_prob):
    p = []
    # 第一块 conv1_1-conv1_2-pool1
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3,
                n_out = 64, dh = 1, dw = 1, p = p)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3,
                n_out = 64, dh = 1, dw = 1, p = p)
    pool1 = mpool_op(conv1_2, name = 'pool1', kh = 2, kw = 2,
                dw = 2, dh = 2)
    # 第二块 conv2_1-conv2_2-pool2
    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3,
                n_out = 128, dh = 1, dw = 1, p = p)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3,
                n_out = 128, dh = 1, dw = 1, p = p)
    pool2 = mpool_op(conv2_2, name = 'pool2', kh = 2, kw = 2,
                dw = 2, dh = 2)
    # 第三块 conv3_1-conv3_2-conv3_3-pool3
    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3,
                n_out = 256, dh = 1, dw = 1, p = p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3,
                n_out = 256, dh = 1, dw = 1, p = p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3,
                n_out = 256, dh = 1, dw = 1, p = p)
    pool3 = mpool_op(conv3_3, name = 'pool3', kh = 2, kw = 2,
                dw = 2, dh = 2)
    # 第四块 conv4_1-conv4_2-conv4_3-pool4
    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3,
                n_out = 512, dh = 1, dw = 1, p = p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3,
                n_out = 512, dh = 1, dw = 1, p = p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3,
                n_out = 512, dh = 1, dw = 1, p = p)
    pool4 = mpool_op(conv4_3, name = 'pool4', kh = 2, kw = 2,
                dw = 2, dh = 2)
    # 第五块 conv5_1-conv5_2-conv5_3-pool5
    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3,
                n_out = 512, dh = 1, dw = 1, p = p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3,
                n_out = 512, dh = 1, dw = 1, p = p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3,
                n_out = 512, dh = 1, dw = 1, p = p)
    pool5 = mpool_op(conv5_3, name = 'pool5', kh = 2, kw = 2,
                dw = 2, dh = 2)
    # 把pool5 ( [7, 7, 512] )  拉成向量
    shp  = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name = 'resh1')

    # 全连接层1 添加了 Droput来防止过拟合    
    fc1 = fc_op(resh1, name = 'fc1', n_out = 2048, p = p)
    fc1_drop = tf.nn.dropout(fc1, keep_prob, name = 'fc1_drop')

    # 全连接层2 添加了 Droput来防止过拟合    
    fc2 = fc_op(fc1_drop, name = 'fc2', n_out = 2048, p = p)
    fc2_drop = tf.nn.dropout(fc2, keep_prob, name = 'fc2_drop')

    # 全连接层3 加一个softmax求给类别的概率
    fc3 = fc_op(fc2_drop, name = 'fc3', n_out = 1000, p = p)
    softmax = tf.nn.softmax(fc3)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc3, p

# 定义评测函数

def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict = feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i  % 10: 
                print('%s: step %d, duration = %.3f' % 
                    (datetime.now(), i-num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mean_dur = total_duration / num_batches 
    var_dur = total_duration_squared / num_batches - mean_dur * mean_dur
    std_dur = math.sqrt(var_dur)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %(datetime.now(), info_string, num_batches, mean_dur, std_dur))


def train_vgg16():
    with tf.Graph().as_default():
        image_size = 224  # 输入图像尺寸
        # 生成随机数测试是否能跑通
        #images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()
        keep_prob = tf.placeholder(tf.float32)
        prediction,softmax,fc8,p = inference_op(images,keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, prediction,{keep_prob:1.0}, "Forward")
        # 用以模拟训练的过程
        objective = tf.nn.l2_loss(fc8)  # 给一个loss
        grad = tf.gradients(objective, p)  # 相对于loss的 所有模型参数的梯度
        time_tensorflow_run(sess, grad, {keep_prob:0.5},"Forward-backward")




if __name__ == '__main__':
    train_vgg16()
