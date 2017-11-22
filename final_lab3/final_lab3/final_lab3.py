# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners



"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pygame
from pygame import *

WIN_WIDTH = 900  # Ширина создаваемого окна
WIN_HEIGHT = 640  # Высота
CELL_WIDTH=20
CELL_HEIGHT=20
DISPLAY = (WIN_WIDTH, WIN_HEIGHT)  # Группируем ширину и высоту в одну переменную
BACKGROUND_COLOR = "#ffffff"


arr=[]
for i in range(28):
    arr.append([])
    for j in range(28):
        arr[len(arr)-1].append(0)


for i in range(28):
    print(i)
print(arr)


arr_saved=arr
our=[]


def main1():
    global arr
    loop = 0
    pygame.init()  # Инициация PyGame, обязательная строчка
    screen = pygame.display.set_mode(DISPLAY)  # Создаем окошко
    pygame.display.set_caption("Введите изображение, потом цифру.После ввода всех цифр - введите режим. Тренировка-белая кнопка справа. Распознавание - синяя")  # Пишем в шапку
    bg = Surface((WIN_WIDTH, WIN_HEIGHT))  # Создание видимой поверхности
    # будем использовать как фон
    bg.fill(Color(BACKGROUND_COLOR))  # Заливаем поверхность сплошным цветом

    while 1:  # Основной цикл программы
        loop+=1
        for e in pygame.event.get():  # Обрабатываем события
            if e.type == QUIT:
                our.append(arr)
                return 0
                pass
            if e.type==MOUSEBUTTONDOWN:
                if e.button == 1:  # левая кнопка мыши
                    x_ch=e.pos[1] //20-1
                    y_ch=e.pos[0]//20-1
                    print(x_ch,y_ch)
                    if x_ch==0 and y_ch==34:
                        print(our)
                        return 1
                    else:
                        if x_ch==4 and y_ch==34:
                            print (our)
                            return 2
                        else:
                            if arr[x_ch][y_ch]==1:
                                arr[x_ch][y_ch]=0
                            else:
                                arr[x_ch][y_ch]=1
                    bg.fill(Color(BACKGROUND_COLOR))
            if e.type==KEYDOWN:
                    pass
                    #our.append([arr,e.key-48])
                    #arr = arr_saved
                    #bg.fill(Color(BACKGROUND_COLOR))
        screen.blit(bg, (0, 0))  # Каждую итерацию необходимо всё перерисовывать
        pygame.display.update()  # обновление и вывод всех изменений на экран
        for X in range(CELL_WIDTH, 570, CELL_WIDTH):
            for Y in range(CELL_HEIGHT, 570, CELL_HEIGHT):
                color=arr[int(Y / 20) - 1][int(X / 20) - 1]
                if color==0:
                    color_real=1
                else:
                    color_real=0
                cell=pygame.draw.rect(bg, (0,255,0), (X, Y, CELL_WIDTH, CELL_HEIGHT),color_real)
                if loop==1:
                    pass
        pygame.draw.rect(bg, (0, 255, 255), (700, 20, CELL_WIDTH, CELL_HEIGHT),2)
        pygame.draw.rect(bg, (0, 255, 255), (700, 100, CELL_WIDTH, CELL_HEIGHT), 0)





import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import sys
import tempfile
import argparse

FLAGS = None

class Capture():
    
    def __init__(self):
        def weight_variable(shape):
              """weight_variable generates a weight variable of a given shape."""
              initial = tf.truncated_normal(shape, stddev=0.1)
              return tf.Variable(initial)
        self.W_conv1=weight_variable([5, 5, 1, 32])
        self.W_conv2=weight_variable([5, 5, 32, 64])
        self.W_fc1= weight_variable([7 * 7 * 64, 1024])
        self.W_fc2=weight_variable([1024, 10])
        
    
    def train(self,_):
          FLAGS = None
          def deepnn(x):
          # Reshape to use within a convolutional neural net.
          # Last dimension is for "features" - there is only one here, since images are
          # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
              with tf.name_scope('reshape'):
                    x_image = tf.reshape(x, [-1, 28, 28, 1])

              # First convolutional layer - maps one grayscale image to 32 feature maps.
              with tf.name_scope('conv1'):
                #W_conv1 = weight_variable([5, 5, 1, 32])
                b_conv1 = bias_variable([32])
                h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + b_conv1)

              # Pooling layer - downsamples by 2X.
              with tf.name_scope('pool1'):
                h_pool1 = max_pool_2x2(h_conv1)

              # Second convolutional layer -- maps 32 feature maps to 64.
              with tf.name_scope('conv2'):
                #W_conv2 = weight_variable([5, 5, 32, 64])
                b_conv2 = bias_variable([64])
                h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + b_conv2)

              # Second pooling layer.
              with tf.name_scope('pool2'):
                h_pool2 = max_pool_2x2(h_conv2)

              # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
              # is down to 7x7x64 feature maps -- maps this to 1024 features.
              with tf.name_scope('fc1'):
                #W_fc1 = weight_variable([7 * 7 * 64, 1024])
                b_fc1 = bias_variable([1024])

                h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + b_fc1)

              # Dropout - controls the complexity of the model, prevents co-adaptation of
              # features.
              with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

              # Map the 1024 features to 10 classes, one for each digit
              with tf.name_scope('fc2'):
                #W_fc2 = weight_variable([1024, 10])
                b_fc2 = bias_variable([10])

                y_conv = tf.matmul(h_fc1_drop, self.W_fc2) + b_fc2
              return y_conv, keep_prob


          def conv2d(x, W):
                """conv2d returns a 2d convolution layer with full stride."""
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


          def max_pool_2x2(x):
                """max_pool_2x2 downsamples a feature map by 2X."""
                return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')


          


          def bias_variable(shape):
              """bias_variable generates a bias variable of a given shape."""
              initial = tf.constant(0.1, shape=shape)
              return tf.Variable(initial)


          def main(_):
              # Import data
              mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

              # Create the model
              x = tf.placeholder(tf.float32, [None, 784])

              # Define loss and optimizer
              y_ = tf.placeholder(tf.float32, [None, 10])

              # Build the graph for the deep net
              y_conv, keep_prob = deepnn(x)

              with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                        logits=y_conv)
              cross_entropy = tf.reduce_mean(cross_entropy)

              with tf.name_scope('adam_optimizer'):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

              with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
              accuracy = tf.reduce_mean(correct_prediction)

              graph_location = tempfile.mkdtemp()
              print('Saving graph to: %s' % graph_location)
              train_writer = tf.summary.FileWriter(graph_location)
              train_writer.add_graph(tf.get_default_graph())

              with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(20000):
                  batch = mnist.train.next_batch(50)
                  if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

                print('test accuracy %g' % accuracy.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

          if __name__ == '__main__':
                  parser = argparse.ArgumentParser()
                  parser.add_argument('--data_dir', type=str,
                                      default='/tmp/tensorflow/mnist/input_data',
                                      help='Directory for storing input data')
                  FLAGS, unparsed = parser.parse_known_args()
                  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    def recognize(self,_):
        pass

Our=Capture()

if __name__ == '__main__':
  if (main1()==1):
      parser = argparse.ArgumentParser()
      parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                          help='Directory for storing input data')
      FLAGS, unparsed = parser.parse_known_args()
      tf.app.run(main=Our.train, argv=[sys.argv[0]] + unparsed)
      main1()
  else:
       pass


