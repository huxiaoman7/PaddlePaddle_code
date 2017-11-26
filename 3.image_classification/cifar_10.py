#coding:utf-8
'''
Created bu huxiaoman on 2017.11.26
cifar_10.py: 用cifar-10数据集来做图像识别
'''

import sys,os
import paddle.v2 as paddle

from vgg import bgg_bn_drop
from resnet import resnet_cifar10

with_gpu = os.getenv('WITH_GPU','0') != '1'



