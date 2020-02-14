import tensorflow as tf

num_class = 3

with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
    Y = tf.placeholder(tf.int32, [None])

    with tf.variable_scope('CNN'):
        net = tf.layers.conv2d(X, 20, 3, (2, 2), padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.conv2d(net, 40, 3, (2, 2), padding='same', activation=tf.nn.relu)
        net = tf.layers.flatten(net)

        out = tf.layers.dense(net, num_class)

    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=out))

    train = tf.train.AdamOptimizer(1e-3).minimize(loss)
    saver = tf.train.Saver()

batch_size = 1461
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30):
        batch_data, batch_label = batch(trainlist, batch_size)
        _, l = sess.run([train, loss], feed_dict={X: batch_data, Y: batch_label})
        print(i, l)

    saver.save(sess, 'logs/model.ckpt', global_step=i + 1)