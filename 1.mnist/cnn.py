#coding:utf-8

import numpy as np
from activators import ReluActivator,IdentityActivator

class ConvLayer(object):
	def __init__(self,input_width,input_weight,
		     channel_number,filter_width,
		     filter_height,filter_number,
		     zero_padding,stride,activator,
		     learning_rate):
		self.input_width = input_width
		self.input_height = input_height
		self.channel_number = channel_number
		self.filter_width = filter_width
		self.filter_height = filter_height
		self.filter_number = filter_number
		self.zero_padding = zero_padding
		self.stride = stride #此处可以加上stride_x, stride_y
		self.output_width = ConvLayer.calculate_output_size(
				self.input_width,filter_width,zero_padding,
				stride)
		self.output_height = ConvLayer.calculate_output_size(
				self.input_height,filter_height,zero_padding,
				stride)
		self.output_array = np.zeros((self.filter_number,
				self.output_height,self.output_width))
		self.filters = []
		for i in range(filter_number):	
			self.filters.append(Filter(filter_width,
				filter_height,self.channel_number))
		self.activator = activator
		self.learning_rate = learning_rate
	def forward(self,input_array):
		'''
		计算卷积层的输出
		输出结果保存在self.output_array
		'''
		self.input_array = input_array
		self.padded_input_array = padding(input_array,
			self.zero_padding)
		for i in range(self.filter_number):
			filter = self.filters[f]
			conv(self.padded_input_array,
			     filter.get_weights(), self.output_array[f],
			     self.stride, filter.get_bias())
			element_wise_op(self.output_array,
					self.activator.forward)

def get_batch(input_array, i, j, filter_width,filter_height,stride):
	'''
	从输入数组中获取本次卷积的区域，
	自动适配输入为2D和3D的情况
	'''
	start_i = i * stride
	start_j = j * stride
	if input_array.ndim == 2:
		return input_array[
			start_i : start_i + filter_height,
			start_j : start_j + filter_width]
	elif input_array.ndim == 3:
		return input_array[
			start_i : start_i + filter_height,
                        start_j : start_j + filter_width]

# 获取一个2D区域的最大值所在的索引
def get_max_index(array):
	max_i = 0
	max_j = 0
	max_value = array[0,0]
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if array[i,j] > max_value:
				max_value = array[i,j]
				max_i, max_j = i, j
	return max_i, max_j

def conv(input_array,kernal_array,
	output_array,stride,bias):
	'''
	计算卷积，自动适配输入2D,3D的情况
	'''
	channel_number = input_array.ndim
	output_width = output_array.shape[1]
	output_height = output_array.shape[0]
	kernel_width = kernel_array.shape[-1]
	kernel_height = kernel_array.shape[-2]
	for i in range(output_height):
		for j in range(output_width):
			output_array[i][j] = (
			    get_patch(input_array, i, j, kernel_width,
			    	kernel_height,stride) * kernel_array).sum() +bias



def element_wise_op(array, op):
	for i in np.nditer(array,
			   op_flags = ['readwrite']):
	    i[...] = op(i)


class ReluActivators(object):
	def forward(self, weighted_input):
		# Relu计算公式 = max(0,input)
		return max(0, weighted_input)

	def backward(self,output):
		return if output > 0 else 0

class SigmoidActivator(object):
		
	def forward(self,weighted_input):
		return 1 / (1 + math.exp(- weighted_input))
	
	def backward(self,output):
		return output * (1 - output)


