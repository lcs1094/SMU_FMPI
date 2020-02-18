import tensorflow as tf


# Cost Function

def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.compat.v1.log(hypothesis) + (1 - labels) * tf.compat.v1.log(1 - hypothesis))
    return cost


# Cost Optimization

def grad(hypothesis, labels):
    with tf.GradientTape as tape:
        loss_value = loss_fn(hypothesis, labels)
    return tape.gradient(loss_value, [W, b])


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
