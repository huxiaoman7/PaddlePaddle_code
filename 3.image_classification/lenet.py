#coding:utf-8
'''
Created by huxiaoman 2017.11.27
lenet.py:LeNet-5网络结构
'''
import os
from PIL import Image
import numpy as np
import paddle.v2 as paddle
from paddle.trainer_config_helpers import *

with_gpu = os.getenv('WITH_GPU', '0') != '1'

def lenet(img):
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=6,
        num_channel=3,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=16,
	num_channel=120,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    conv_3 = img_conv_layer(
        input = conv_pool_2,
        filter_size = 1,
        num_filters = 120,
        stride = 1)
    fc = paddle.layer.fc(
        input=conv_3, size=512, act=paddle.activation.Softmax())
    return fc
