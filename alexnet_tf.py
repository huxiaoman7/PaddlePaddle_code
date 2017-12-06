#coding:utf-8
'''
Created by huxiaoman 2017.12.05
alexnet_tf.py:训练tensorflo版的alexnet网络
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

FLAGS = None


def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
  '''
  Alexnet模型
  输入：images的tensor
  返回：Alexnet的最后一层卷积层

  '''
  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]

  # lrn1
  with tf.name_scope('lrn1') as scope:
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool1
  pool1 = tf.nn.max_pool(lrn1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
  print_activations(pool1)

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(conv2)

  # lrn2
  with tf.name_scope('lrn2') as scope:
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool2
  pool2 = tf.nn.max_pool(lrn2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
  print_activations(pool2)

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
  print_activations(pool5)

  return pool5, parameters


def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))



def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default():
    image_size = 224
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    pool5, parameters = inference(images)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    sess.run(init)

    time_tensorflow_run(sess, pool5, "Forward")

    objective = tf.nn.l2_loss(pool5)
    grad = tf.gradients(objective, parameters)
    time_tensorflow_run(sess, grad, "Forward-backward")


def main(_):
  run_benchmark()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size.'
  )
  parser.add_argument(
      '--num_batches',
      type=int,
      default=200,
      help='Number of batches to run.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
