#coding:utf-8
'''
Created by huxiaoman 2017.10.28
Copyright huxiaoman 
mnist.py:using paddlepaddle framework to train a simple cnn network,and improve the baseline network,the accuracy of improved network is 99.28% 
'''
import os
from PIL import Image
import numpy as np
import paddle.v2 as paddle

# 设置是否用gpu，0为否，1为是
with_gpu = os.getenv('WITH_GPU', '0') != '1'

# 定义网络结构
def convolutional_neural_network_org(img):
    # 第一层卷积层
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # 第二层卷积层
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # 全连接层
    predict = paddle.layer.fc(
        input=conv_pool_2, size=10, act=paddle.activation.Softmax())
    return predict

# 改进版网络结构
def convolutional_neural_network(img):
    # 第一层卷积层
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # 加一层dropout层
    drop_1 = paddle.layer.dropout(input=conv_pool_1, dropout_rate=0.2)
    # 第二层卷积层
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=drop_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # 加一层dropout层
    drop_2 = paddle.layer.dropout(input=conv_pool_2, dropout_rate=0.5)
    # 全连接层
    fc1 = paddle.layer.fc(input=drop_2, size=10, act=paddle.activation.Linear())
    bn = paddle.layer.batch_norm(input=fc1,act=paddle.activation.Relu(),
         layer_attr=paddle.attr.Extra(drop_rate=0.2))
    predict = paddle.layer.fc(input=bn, size=10, act=paddle.activation.Softmax())
    return predict


def main():
    # 初始化定义跑模型的设备
    paddle.init(use_gpu=with_gpu, trainer_count=1)

    # 读取数据
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))

    # 调用之前定义的网络结构
    predict = convolutional_neural_network_org(images)#原网络结构    
    # predict = convolutional_neural_network(images)

    # 定义损失函数
    cost = paddle.layer.classification_cost(input=predict, label=label)

    # 指定训练相关的参数
    parameters = paddle.parameters.create(cost)

    # 定义训练方法
    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1 / 128.0,
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

    # 训练模型
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)


    lists = []

    # 定义event_handler，输出训练过程中的结果
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        if isinstance(event, paddle.event.EndPass):
            # 保存参数
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                parameters.to_tar(f)

            result = trainer.test(reader=paddle.batch(
                paddle.dataset.mnist.test(), batch_size=128))
            print "Test with Pass %d, Cost %f, %s\n" % (
                event.pass_id, result.cost, result.metrics)
            lists.append((event.pass_id, result.cost,
                          result.metrics['classification_error_evaluator']))

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=64),
        event_handler=event_handler,
        num_passes=50)

    # 找到训练误差最小的一次结果
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
    print 'The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100)

    # 加载数据   
    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = np.array(im).astype(np.float32).flatten()
        im = im / 255.0
        return im

    # 测试结果
    test_data = []
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_data.append((load_image(cur_dir + '/image/infer_3.png'), ))

    probs = paddle.infer(
        output_layer=predict, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)  # probs and lab are the results of one batch data
    print "Label of image/infer_3.png is: %d" % lab[0][0]


if __name__ == '__main__':
    main()
