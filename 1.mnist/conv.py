#coding:utf-8
'''
Created by huxiaoman 2017.10.31
Update by huxiaoman 2017.11.15
Copyright huxiaoman
conv.py:to implement a simple convolutional network,includig convolutional,padding,maxpolling,forwarf_propogation and backpropogation process.
###暂时不要用这个程序，写的有点问题，输出结果不确定是对的 = =
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
    f:feature,卷积核数目
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
	self.w_num = (w - k_x + 2 * p_x) / s_x + 1
	self.h_num = (h - k_y + 2 * p_y) / s_y + 1
	self.weights = np.random.randn(f, c * k_x * k_y) / np.sqrt(c * k_x * k_y)
	self.bias = np.random.randn(f)
	self.lr,self.lamb = 0.0, 0.0

    # zero padding
    def padding(self, x):
	zero_num = x.shape
	zero = np.zeros((zero_num[0], zero_num[1], zero_num[2] + 2* self.p_x, zero_num[3] + 2* self.p_y))
	x = zero[:, :, self.p_x:self.p_x + self.w, self.p_y:self.p_y + self.h]
	return zero

    def eval(self,x):
	# 补零后的宽度ww和高度hh
	ww = self.h - self.k_x + 2 * self.p_x + 1
	hh = self.h - self.k_y + 2 * self.p_y + 1
	ret = np.array([[[np.ravel(xx[:, a:a + self.k_x, b:b + self.k_y]) for b in range(0, hh, self.s_y)]
                         for a in range(0, ww, self.s_x)] for xx in x])
	#ret = np.array([[[np.ravel(xx[:,a:a +self.k_x, b:b, self.k_y]) for b in range(0,hh,self.s_y)] for a in range(0,ww,self.s_x)] for xx in x])# here using np.ravel rather than np.flatten to save memory
	return ret
    
    def de_eval(self,x):
	a1, a2, a3, a4 = x.shape
        x = x.reshape(a1, a2, a3, self.c, self.k_x, self.k_y)
        ret = np.zeros_like(self.px)
        for i in range(a1):
            for j in range(a2):
                w = j * self.s_x
                for t in range(a3):
                    h = t * self.s_y
                    ret[i][:, w:w + self.k_x, h:h + self.k_y] += x[i][j][t]
        return ret[:, :, self.p_x:self.p_x + self.w, self.p_y:self.p_y + self.h]


    def forward(self,x):
	assert (x.shape[1] == self.c)
	assert (x.shape[2] == self.w)
	assert (x.shape[3] == self.h)
	self.px = self.padding(x)
	self.evalx = self.eval(self.px)
	self.y = (self.evalx.dot(self.weights.T) + self.bias).transpose(0,3,1,2)
	print self. y	
	return self.y

    def backward(self,b):
        a1, a2, a3, a4 = b.shape
	self.ddx = np.array([dd.T.dot(self.weights) for dd in b])
        self.dx = self.de_eval(self.ddx)
        b = b.reshape(a1, a2, a3 * a4)
        self.evalx = self.evalx.reshape(a1, a3 * a4, self.c * self.k_x * self.k_y)
        self.dw = np.sum([dd.dot(x) for x, dd in zip(self.evalx, b)], axis=0) / a1 / a3 / a4
        self.db = np.sum(b, axis=(0, 2)) / a1 / a3 / a4

        self.weights -= self.lr * (self.dw + self.lamb * np.sum(np.square(self.weights)) / a1)
        self.bias -= self.lr * self.db
	return self.dx

if __name__ == '__main__':
	# 带入数值计算
	test = Conv(1,5,5,1,1,1,1,3,3,32)
	data = np.ones((6,1,5,5))
	# 前向传播
	x1 = test.forward(data)
	# 反向传播
	x2 = test.backward(x1)
	print x1.shape
	print x2.shape

	
