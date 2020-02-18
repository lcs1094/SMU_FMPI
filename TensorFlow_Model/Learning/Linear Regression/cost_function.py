import numpy as np
import tensorflow as tf

# cost_function in Python
# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])
#
#
# def cost_func(W, X, Y):
#     c = 0
#     for i in range(len(X)):
#         c += (W * X[i] - Y[i]) ** 2
#     return c / len(X)
#
#
# for feed_W in np.linspace(-3, 5, num=15):
#     curr_cost = cost_func(feed_W, X, Y)
#     print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

# cost_function in Tensorflow
X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])


def cost_func(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))


W_values = np.linspace(-3, 5, num=15)
cost_values = []
for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))