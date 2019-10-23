import tensorflow as tf

SAVE_PATH = "mypath.ckpt"

def saver(W_in,b_in):
    W = tf.Variable(W_in,dtype=tf.float32, name='weights')
    b = tf.Variable(b_in,dtype=tf.float32,name='biases')

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, SAVE_PATH)
        print("Save to path: ", save_path)
