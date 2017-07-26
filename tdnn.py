import tensorflow as tf
import numpy as np

tf.set_random_seed(922)
np.random.seed(0)

def tdnn(inp, n_class):
    h1_kernel_size = 3
    h1_filters = 8
    h1 = tf.layers.conv1d(inp, h1_filters, h1_kernel_size)
    h2_kernel_size = 5
    h2_filters = n_class
    h2 = tf.layers.conv1d(h1, h2_filters, h2_kernel_size)
    h2 = tf.transpose(h2, [0, 2, 1])
    output = tf.squeeze(tf.layers.dense(h2, 1, tf.sigmoid))
    return output 

if __name__ == '__main__':
    bs = 1
    n_frame = 15
    n_feature = 16
    n_class = 3
    inp=tf.placeholder(tf.float32, shape=[bs, n_frame, n_feature])
    output = tdnn(inp, n_class)
    with tf.Session() as sess:
        i = np.random.normal(size=(bs, n_frame, n_feature)) 
        sess.run(tf.global_variables_initializer())
        o = sess.run(output, feed_dict={inp: i})
        print np.array(o).shape  # (3,)
        print np.array(o) # [ 0.64834243  0.91836888  0.50499392] 
