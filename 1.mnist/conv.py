#coding:utf-8
'''
Created by huxiaoman 2017.10.31
Copyright huxiaoman
conv.py:to implement a simple convolutional network,includig convolutional,padding,maxpolling,forwarf_propogation and backpropogation process.
'''

import numpy as np
class Conv:
    '''
    参数含义：
    c:channel,通道数
    w:width,图片的宽度
    h:height,图片长度
    k_x:kernel_x,卷积核长度
    k_y:kernel_y,卷积核宽度
    s_x:stride_x,水平步长长度
    s_y:stride_y,垂直步长长度
    p_x:zero_padding_x,水平补零长度
    p_y:zero_padding_y,垂直补零长度
    f:feature,特征数目
    '''
    def __init__(self, c, w, h, k_x, k_y, s_x, s_y, p_x, p_y, f):
	self.c, self.w, self.h = c, w, h
	self.k_x, self.k_y = k_x,k_y
        self.s_x, self.s_y = s_x, s_y
	self.p_x, self.p_y = p_x, p_y
	self.f = f
	# 判断水平方向上的卷积层输出的神经元个数为整数
	assert ((w - k_x + 2 * p_x) % s_x ==0)
	# 判断垂直方向上的卷积层输出的神经元个数为整数
	assert ((h - k_y + 2 * p_y) % s_y ==0)
	self.w_num = (w - k_x + 2 * p_x) % s_x) + 1
	self.h_num = (h - k_y + 2 * p_y) % s_y) + 1
	self.weights = np.random.randn(f, c * k_x * k_y) / nq.sqrt(c * k_x * k_y)
	self.bias = np.random.randn(f)
	self.lr,self.lamb = 0.0, 0.0

    # zero padding
    def padding(self, x):
	zero_num = x.shape
	zero = np.zeros((zero_num[0], zero_num[1], zero_num[2] + 2* self.p_x, zero_num[3] + 2* self.p_y))
	x = zero[:, :, self.p_x:self.p_x + self.w, self.p_y:self.p_y + self.h]
	return zero

    def forward(self,x):
	assert (x.shape[1] == self.c)
	assert (x.shape[2] == self.w)
	assert (x.shape[3] == self.h)
	self.px = self.pad(x)
	self.evalx = self.eval(self.px)
	self.y = (self.evalx.dot(self.weights.T) + self.bias).transpose(0,3,1,2)
	
	return self.y

	
