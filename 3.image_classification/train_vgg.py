#coding:utf-8
'''
Created by huxiaoman 2017.12.12
train_vgg.py:训练vgg16对cifar10数据集进行分类
'''

import sys, os
import paddle.v2 as paddle
from vggnet import vgg

with_gpu = os.getenv('WITH_GPU', '0') != '1'

def main():
    datadim = 3 * 32 * 32
    classdim = 10

    # PaddlePaddle init
    paddle.init(use_gpu=with_gpu, trainer_count=8)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(datadim))
    
    net = vgg(image)

    out = paddle.layer.fc(
        input=net, size=classdim, act=paddle.activation.Softmax())

    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(classdim))
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # Create parameters
    parameters = paddle.parameters.create(cost)

    # Create optimizer
    momentum_optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
        learning_rate=0.1 / 128.0,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=50000 * 100,
        learning_rate_schedule='discexp')

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                parameters.to_tar(f)

            result = trainer.test(
                reader=paddle.batch(
                    paddle.dataset.cifar.test10(), batch_size=128),
                feeding={'image': 0,
                         'label': 1})
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    # Create trainer
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=momentum_optimizer)

    # Save the inference topology to protobuf.
    inference_topology = paddle.topology.Topology(layers=out)
    with open("inference_topology.pkl", 'wb') as f:
        inference_topology.serialize_for_inference(f)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.cifar.train10(), buf_size=50000),
            batch_size=128),
        num_passes=200,
        event_handler=event_handler,
        feeding={'image': 0,
                 'label': 1})

    # inference
    from PIL import Image
    import numpy as np
    import os

    def load_image(file):
        im = Image.open(file)
        im = im.resize((32, 32), Image.ANTIALIAS)
        im = np.array(im).astype(np.float32)
        im = im.transpose((2, 0, 1))  # CHW
        im = im[(2, 1, 0), :, :]  # BGR
        im = im.flatten()
        im = im / 255.0
        return im

    test_data = []
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_data.append((load_image(cur_dir + '/image/dog.png'), ))

    probs = paddle.infer(
        output_layer=out, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)  # probs and lab are the results of one batch data
    print "Label of image/dog.png is: %d" % lab[0][0]


if __name__ == '__main__':
    main()
