#/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np
import random

#读取文件数据
def reader(filename):
    data = []
    f = open(filename,'r')
    for line in f.readlines():
        row = line.strip('\n').split('\t')
        data.append([float(i) for i in row])
    data = np.array(data)
    f.close()
    return data

# 生成batch数据
def batch(data,batch_size):
    x = []
    for i in data:
        x.append(i)
        if len(x) == batch_size:
            yield x
            x = []
    if x:
        yield x

# 生成内置batch数据
def input(data,batch_size):
    #count = len(data)/batch_size
    #input_data = np.array(list(batch(data,batch_size)))[random.randint(0,count)]
    input_data = np.array(list(batch(data,batch_size)))
    return input_data

# 生成文件batch数据
def file_input(filename,batch_size):
    data = reader(filename)
    input_data = input(data,batch_size)
    return input_data


class AutoEncoder():
    '''
    Auto Encoder  
    layer      1     2    ...    ...    L-1    L
      W        0     1    ...    ...    L-2
      B        0     1    ...    ...    L-2
      Z              0     1     ...    L-3    L-2
      A              0     1     ...    L-3    L-2
    '''
    def __init__(self, X, Y, nNodes):
        # training samples
        self.X = X
        self.Y = Y
        # number of samples
        self.M = len(self.X)
        # layers of networks
        self.nLayers = len(nNodes)
        # nodes at layers
        self.nNodes = nNodes
        # parameters of networks
        self.W = list()
        self.B = list()
        self.dW = list()
        self.dB = list()
        self.A = list()
        self.Z = list()
        self.delta = list()
        for iLayer in range(self.nLayers - 1):
            self.W.append( np.random.rand(nNodes[iLayer]*nNodes[iLayer+1]).reshape(nNodes[iLayer],nNodes[iLayer+1]) ) 
            self.B.append( np.random.rand(nNodes[iLayer+1]) )
            self.dW.append( np.zeros([nNodes[iLayer], nNodes[iLayer+1]]) )
            self.dB.append( np.zeros(nNodes[iLayer+1]) )
            self.A.append( np.zeros(nNodes[iLayer+1]) )
            self.Z.append( np.zeros(nNodes[iLayer+1]) )
            self.delta.append( np.zeros(nNodes[iLayer+1]) )
            
        # value of cost function
        self.Jw = 0.0
        # active function (logistic function)
        self.sigmod = lambda z: 1.0 / (1.0 + np.exp(-z))
        # learning rate 1.2
        self.alpha = 2.5
        # steps of iteration 30000
        self.steps = 100
        
    def BackPropAlgorithm(self):
        # 定义loss方式
        self.Jw -= self.Jw
        for iLayer in range(self.nLayers-1):
            self.dW[iLayer] -= self.dW[iLayer]
            self.dB[iLayer] -= self.dB[iLayer]
        # 前向传播和反向传播 
        for i in range(self.M):
            # 前向传播
            for iLayer in range(self.nLayers - 1):
                # 第一层
                if iLayer==0: 
                    self.Z[iLayer] = np.dot(self.X[i], self.W[iLayer])
                else:
                    self.Z[iLayer] = np.dot(self.A[iLayer-1], self.W[iLayer])
                self.A[iLayer] = self.sigmod(self.Z[iLayer] + self.B[iLayer])            
            # 反向传播
            for iLayer in range(self.nLayers - 1)[::-1]: # reserve
                if iLayer==self.nLayers-2:# 最后一层
                    self.delta[iLayer] = -(self.X[i] - self.A[iLayer]) * (self.A[iLayer]*(1-self.A[iLayer]))#sigmoid的导数
                    self.Jw += np.dot(self.Y[i] - self.A[iLayer], self.Y[i] - self.A[iLayer])/self.M
                else:
                    self.delta[iLayer] = np.dot(self.W[iLayer].T, self.delta[iLayer+1]) * (self.A[iLayer]*(1-self.A[iLayer]))
                # 计算权值和偏置的偏导
                if iLayer==0:
                    self.dW[iLayer] += self.X[i][:, np.newaxis] * self.delta[iLayer][:, np.newaxis].T
                else:
                    self.dW[iLayer] += self.A[iLayer-1][:, np.newaxis] * self.delta[iLayer][:, np.newaxis].T
                self.dB[iLayer] += self.delta[iLayer] 
        # 更新权重和偏置
        for iLayer in range(self.nLayers-1):
            self.W[iLayer] -= (self.alpha/self.M)*self.dW[iLayer]
            self.B[iLayer] -= (self.alpha/self.M)*self.dB[iLayer]

        return self.W,self.B

    
    # AutoEncoder计算过程，打印loss值
    def PlainAutoEncoder(self):
        for i in range(self.steps):
            self.BackPropAlgorithm()
            print "step:%d" % i, "loss=%f" % self.Jw

    # 输出每层的神经元信息
    def ValidateAutoEncoder(self):
        for i in range(self.M):
            print self.X[i]
            for iLayer in range(self.nLayers - 1):
                if iLayer==0: 
                    self.Z[iLayer] = np.dot(self.X[i], self.W[iLayer])
                else:
                    self.Z[iLayer] = np.dot(self.A[iLayer-1], self.W[iLayer])
                self.A[iLayer] = self.sigmod(self.Z[iLayer] + self.B[iLayer])
                print "\t layer=%d" % iLayer, self.A[iLayer]

def batch_train(data,batch_size):
    input_data = input(data,batch_size)
    count  = len(data)/batch_size
    nNodes =  np.array([ 200, 10, 200])
    for i in range(count):
        x = input_data[i]
        attr = AutoEncoder(data,data,nNodes)
        print "第%d个batch" %i, attr.PlainAutoEncoder()

def main(filename):
    file = reader(filename)
    data = batch_reader(file)
    ae2 = AutoEncoder(data,data,nNodes)
    ae2.PlainAutoEncoder()
    ae2.ValidateAutoEncoder()

if __name__=='__main__':
    # 内置生成的array数据
    raw_data = np.random.random(size=(1000,200))
    # data = input(raw_data,100)
    # 读取文件数据
    # data = file_input('data.txt',100)
    batch_train(raw_data,100)
    nNodes =  np.array([ 200, 10, 200])
    # ae2 = AutoEncoder(data,data,nNodes)
    # ae2.PlainAutoEncoder()
    # ae2.ValidateAutoEncoder()    

    # 外部读取文件处理
    #main(sys.argv[1])
    #data_1 = reader(data)
    #data_1 = np.random.random(size=(100000,2000))


