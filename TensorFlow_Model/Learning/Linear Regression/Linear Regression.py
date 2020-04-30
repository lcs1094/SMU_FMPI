import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# reduce_mean -> 산술평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step%20==0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# # 2버전에서 어떻게 바꾸는지 모르겠음 ㅎㅎ
# x_train = [1,2,3]
# y_train = [1,2,3]
#
# W = tf.Variable(tf.compat.v1.random_normal([1]), name = 'weight')
# b = tf.Variable(tf.compat.v1.random_normal([1]), name = 'bias')
#
# hypothesis = x_train * W + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# # reduce_mean -> 산술평균
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
# for step in range(2001):
#     hypothesis = x_train * W + b
#     cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#     optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
#     train = optimizer.minimize(cost)
#     if step % 20 == 0:
#         print(step, cost, W, b)