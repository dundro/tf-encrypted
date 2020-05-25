import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import random as ran

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

sess = tf.Session()


def TRAIN_SIZE(num):
    print('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
    print('--------------------------------------------------')
    x_train = mnist.train.images[:num, :]
    print('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num, :]
    print('y_train Examples Loaded = ' + str(y_train.shape))
    print('')
    return x_train, y_train


def TEST_SIZE(num):
    print('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    print('--------------------------------------------------')
    x_test = mnist.test.images[:num, :]
    print('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num, :]
    print('y_test Examples Loaded = ' + str(y_test.shape))
    return x_test, y_test


def display_digit(num):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28, 28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


def display_mult_flat(start, stop):
    images = x_train[start].reshape([1, 784])
    for i in range(start + 1, stop):
        images = np.concatenate((images, x_train[i].reshape([1, 784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()


x_train, y_train = TRAIN_SIZE(55000)

display_digit(ran.randint(0, x_train.shape[0]))

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros[10])


