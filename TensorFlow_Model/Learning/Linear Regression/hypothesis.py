import tensorflow as tf
# tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution()
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]
W = tf.Variable(2.9)
b = tf.Variable(0.5)

learning_rate = 0.01

for i in range(100):
    # Gradient Descent
    with tf.GradientTape() as tape:
        # 예측값 (가설) : W * x_data + b
        hypothesis = W * x_data + b

        # cos(비용) : 1/m * (1 to m)(H(x)i - yi)^2
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        # tensorflow's reduce_min : average // 차원을 하나 낮춘다
        # tensorflow's square : square

    W_grad, b_grad = tape.gradient(cost, [W, b])
    # gradient : 기울기, 미분

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i%10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))
print(W * 5 + b)
print(W * 2.5 + b)