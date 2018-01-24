#coding:utf-8
'''
Created by huxiaoman 2018.1.23
keras_model_visualization.py:用keras可视化模型训练的过程
'''

from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
import cv2
from cv2 import *
import matplotlib.pyplot as plt
import scipy as sp
from scipy.misc import toimage

def test_opencv():
    # 加载摄像头
    cam = VideoCapture(0)  # 0 -> 摄像头序号，如果有两个三个四个摄像头，要调用哪一个数字往上加嘛
    # 抓拍 5 张小图片
    for x in range(0, 5):
        s, img = cam.read()
        if s:
            imwrite("o-" + str(x) + ".jpg", img)

def load_original(img_path):
    # 把原始图片压缩为 299*299大小
    im_original = cv2.resize(cv2.imread(img_path), (299, 299))
    im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
    plt.figure(0)
    plt.subplot(211)
    plt.imshow(im_converted)
    return im_original

def load_fine_tune_googlenet_v3(img):
    # 加载fine-tuning googlenet v3模型，并做预测
    model = InceptionV3(include_top=True, weights='imagenet')
    model.summary()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
    plt.subplot(212)
    plt.plot(preds.ravel())
    plt.show()
    return model, x

def extract_features(ins, layer_id, filters, layer_num):
    '''
    提取指定模型指定层指定数目的feature map并输出到一幅图上.
    :param ins: 模型实例
    :param layer_id: 提取指定层特征
    :param filters: 每层提取的feature map数
    :param layer_num: 一共提取多少层feature map
    :return: None
    '''
    if len(ins) != 2:
        print('parameter error:(model, instance)')
        return None
    model = ins[0]
    x = ins[1]
    if type(layer_id) == type(1):
        model_extractfeatures = Model(input=model.input, output=model.get_layer(index=layer_id).output)
    else:
        model_extractfeatures = Model(input=model.input, output=model.get_layer(name=layer_id).output)
    fc2_features = model_extractfeatures.predict(x)
    if filters > len(fc2_features[0][0][0]):
        print('layer number error.', len(fc2_features[0][0][0]),',',filters)
        return None
    for i in range(filters):
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.subplot(filters, layer_num, layer_id + 1 + i * layer_num)
        plt.axis("off")
        if i < len(fc2_features[0][0][0]):
            plt.imshow(fc2_features[0, :, :, i])

# 层数、模型、卷积核数
def extract_features_batch(layer_num, model, filters):
    '''
    批量提取特征
    :param layer_num: 层数
    :param model: 模型
    :param filters: feature map数
    :return: None
    '''
    plt.figure(figsize=(filters, layer_num))
    plt.subplot(filters, layer_num, 1)
    for i in range(layer_num):
        extract_features(model, i, filters, layer_num)
    plt.savefig('sample.jpg')
    plt.show()

def extract_features_with_layers(layers_extract):
    '''
    提取hypercolumn并可视化.
    :param layers_extract: 指定层列表
    :return: None
    '''
    hc = extract_hypercolumn(x[0], layers_extract, x[1])
    ave = np.average(hc.transpose(1, 2, 0), axis=2)
    plt.imshow(ave)
    plt.show()

def extract_hypercolumn(model, layer_indexes, instance):
    '''
    提取指定模型指定层的hypercolumn向量
    :param model: 模型
    :param layer_indexes: 层id
    :param instance: 模型
    :return:
    '''
    feature_maps = []
    for i in layer_indexes:
        feature_maps.append(Model(input=model.input, output=model.get_layer(index=i).output).predict(instance))
    hypercolumns = []
    for convmap in feature_maps:
        for i in convmap[0][0][0]:
            upscaled = sp.misc.imresize(convmap[0, :, :, i], size=(299, 299), mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)

if __name__ == '__main__':
    img_path = '~/auto1.jpg'
    img = load_original(img_path)
    x = load_fine_tune_googlenet_v3(img)
    extract_features_batch(15, x, 3)
    extract_features_with_layers([1, 4, 7])
    extract_features_with_layers([1, 4, 7, 10, 11, 14, 17])
