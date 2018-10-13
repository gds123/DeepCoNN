import tensorflow as tf


class DeepCoNN(object):
    def __init__(
            self, user_length, item_length, num_classes, user_vocab_size, item_vocab_size, fm_k, n_latent, user_num,
            item_num,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, l2_reg_V=0.0):
        self.input_u = tf.placeholder(tf.int32, [None, user_length], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        batch_size = tf.shape(self.input_u)[0]
        print("user_length: ", user_length)
        print("item_length: ", item_length)
        print("batch_size ", batch_size)

        l2_loss = tf.constant(0.0)

        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_users = tf.nn.embedding_lookup(self.W1, self.input_u)
            # N x user_len x emb_size (N x in_w x in_c)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_items = tf.nn.embedding_lookup(self.W2, self.input_i)

        x = self.embedded_users
        print("embedded_users: ", x)
        for i in range(len(filter_sizes)):
            k = filter_sizes[i]
            n_channels = num_filters[i]
            with tf.name_scope("user_conv-maxpool-%s" % k):
                conv = tf.layers.conv1d(inputs=x, filters=n_channels, kernel_size=k, strides=1,
                                         padding='same', activation=tf.nn.relu)
                max_pool = tf.layers.max_pooling1d(inputs=conv, pool_size=4, strides=4, padding='same')
                x = max_pool
                print("conv: ", conv)
                print("max_pool: ", max_pool)
        dim = x.get_shape()[1] * x.get_shape()[2]
        self.h_pool_flat_u = tf.reshape(x, [-1, dim])

        x = self.embedded_items
        print("embedded_items: ", x)
        for i in range(len(filter_sizes)):
            k = filter_sizes[i]
            n_channels = num_filters[i]
            with tf.name_scope("item_conv-maxpool-%s" % k):
                conv = tf.layers.conv1d(inputs=x, filters=n_channels, kernel_size=k, strides=1,
                                         padding='same', activation=tf.nn.relu)
                max_pool = tf.layers.max_pooling1d(inputs=conv, pool_size=4, strides=4, padding='same')
                x = max_pool
                print("conv: ", conv)
                print("max_pool: ", max_pool)

        dim = x.get_shape()[1] * x.get_shape()[2]
        self.h_pool_flat_i = tf.reshape(x, [-1, dim])

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)
            print("self.h_drop_u: ", self.h_drop_u)
            print("self.h_drop_i: ", self.h_drop_i)

        with tf.name_scope("get_fea"):
            self.u_fea = tf.layers.dense(self.h_drop_u, n_latent)
            self.i_fea = tf.layers.dense(self.h_drop_i, n_latent)

        with tf.name_scope('fm'):
            self.z = tf.nn.relu(tf.concat([self.u_fea, self.i_fea], axis=1))

            # self.z=tf.nn.dropout(self.z,self.dropout_keep_prob)

            WF1 = tf.Variable(
                tf.random_uniform([n_latent * 2, 1], -0.1, 0.1), name='fm1')
            Wf2 = tf.Variable(
                tf.random_uniform([n_latent * 2, fm_k], -0.1, 0.1), name='fm2')
            one = tf.matmul(self.z, WF1)

            inte1 = tf.matmul(self.z, Wf2)
            inte2 = tf.matmul(tf.square(self.z), tf.square(Wf2))

            inter = (tf.square(inte1) - inte2) * 0.5

            inter = tf.nn.dropout(inter, self.dropout_keep_prob)

            inter = tf.reduce_sum(inter, 1, keepdims=True)
            print(inter)
            b = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = one + inter + b

            print(self.predictions)
        with tf.name_scope("loss"):
            # losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))
