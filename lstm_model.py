import tensorflow as tf
import re


class LSTM_config(object):
    max_document_length = 600
    num_class = 10
    embedding_size = 64
    lstm_size_each_layer = '256, 128'
    use_bidirectional = 0
    use_basic_cell = 1
    use_attention = 1
    attention_size = 200
    grad_clip = 5.0
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32
    dropout_keep_prob = 0.5
    save_per_batch = 10
    print_per_batch = 128


class Lstm(object):
    def __init__(self, config):
        tf.set_random_seed(66)
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_document_length],name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_class],name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.truncated_normal([self.config.vocab_size, self.config.embedding_size],stddev=0.1), name='W')
            self.embedding = tf.nn.embedding_lookup(self.W,self.input_x)

        with tf.variable_scope('layer'):
            lstm_size_each_layer = self.config.lstm_size_each_layer.split(',')
            use_bidirectional = self.config.use_bidirectional
            self.lstm_cell = self.construct_model(lstm_size_each_layer)
            if use_bidirectional:
                self.lstm_cell_bk = self.construct_model(lstm_size_each_layer)
                self.outputs, self.output_states = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell, self.lstm_cell_bk,
                                                                                   self.embedding, dtype=tf.float32)
            else:
                self.outputs, self.output_states = tf.nn.dynamic_rnn(self.lstm_cell, self.embedding,
                                                                     dtype=tf.float32)

        takeall = False
        if 'True' == takeall:
            inputsize_batch = self.args.max_document_lenth * int(lstm_size_each_layer[-1])
        else:
            inputsize_batch = int(lstm_size_each_layer[-1])
        if self.config.use_attention:
            with tf.name_scope('attention'), tf.variable_scope('attention'):
                attention_size = self.config.attention_size
                attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
                u_list = []
                if use_bidirectional:
                    for index in range(2):
                        attention_w = tf.Variable(tf.truncated_normal([inputsize_batch, attention_size], stddev=0.1),
                                                  name='attention_w')
                        if 'True' == takeall:
                            self.outputs_flat = tf.reshape(self.outputs[index], [-1, inputsize_batch])
                        else:
                            self.outputs_flat = tf.unstack(tf.transpose(self.outputs[index], [1, 0, 2]))[-1]
                        u_list.append(tf.tanh(tf.matmul(self.outputs_flat, attention_w) + attention_b))
                else:
                    attention_w = tf.Variable(tf.truncated_normal([inputsize_batch, attention_size], stddev=0.1),
                                              name='attention_w')
                    if 'True' == takeall:
                        self.outputs_flat = tf.reshape(self.outputs, [-1, inputsize_batch])
                    else:
                        self.outputs_flat = tf.unstack(tf.transpose(self.outputs, [1, 0, 2]))[-1]
                    u_list.append(
                        tf.tanh(tf.matmul(self.outputs_flat, attention_w) + attention_b))  # (?, 122, attention_size)

                attn_z = []
                u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
                for index in range(len(u_list)):
                    z_t = tf.matmul(u_list[index], u_w)
                    attn_z.append(z_t)
                attn_zconcat = tf.concat(attn_z, axis=1)
                self.alpha = tf.nn.softmax(attn_zconcat)
                if use_bidirectional:
                    self.alpha = tf.reshape(self.alpha, [-1, 2])
                final_output_tmp = []
                for index in range(len(u_list)):
                    if use_bidirectional:
                        if 'True' == takeall:
                            self.outputs_flat = tf.reshape(self.outputs[index], [-1, inputsize_batch])
                        else:
                            self.outputs_flat = tf.unstack(tf.transpose(self.outputs[index], [1, 0, 2]))[-1]
                        final_output_tmp.append(self.outputs_flat * (tf.reshape(self.alpha[:, index], [-1, 1])))  #
                    else:
                        if 'True' == takeall:
                            self.outputs_flat = tf.reshape(self.outputs, [-1, inputsize_batch])
                        else:
                            self.outputs_flat = tf.unstack(tf.transpose(self.outputs, [1, 0, 2]))[-1]
                        self.final_output = self.outputs_flat * self.alpha
                if use_bidirectional:
                    self.final_output = tf.concat(final_output_tmp, axis=1)
        else:
            final_output_tmp = []
            if use_bidirectional:
                for index in range(2):
                    if 'True' == takeall:
                        final_output_tmp.append(tf.reshape(self.outputs[index], [-1, inputsize_batch]))
                    else:
                        final_output_tmp.append(tf.unstack(tf.transpose(self.outputs[index], [1, 0, 2]))[-1])
            else:
                if 'True' == takeall:
                    self.final_output = tf.reshape(self.outputs, [-1, inputsize_batch])
                else:
                    self.final_output = tf.unstack(tf.transpose(self.outputs, [1, 0, 2]))[-1]
            if use_bidirectional:
                self.final_output = tf.concat(final_output_tmp, axis=1)

        # full connection layer
        with tf.name_scope("output"):
            real_size = inputsize_batch
            if use_bidirectional:
                real_size *= 2
            fc_w = tf.Variable(tf.truncated_normal([real_size, self.config.num_class], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([self.config.num_class]), name='fc_b')
            self.logits = tf.nn.xw_plus_b(self.final_output, fc_w, fc_b, name="logits")
            self.scores = tf.nn.softmax(self.logits)  # 每个文本的问题是某类的得分
            self.predictions = tf.argmax(self.logits, 1, name="predictions")  # 最大得分的index

        # loss
        with tf.name_scope("loss"):
            if 5 <= int(re.findall('\.(.*)?\.', tf.__version__)[0]):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            else:
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))  # 准确率
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        # Optimizer
        with tf.name_scope("Optimizer"):
            """
            打印tensorflow变量的函数有两个：
            tf.trainable_variables () 和 tf.all_variables()
            不同的是：
            tf.trainable_variables () 指的是需要训练的变量
            tf.all_variables() 指的是所有变量
            """
            tvars = tf.trainable_variables()
            # 防止梯度爆炸  对所有参数进行求导
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                              self.config.grad_clip)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.optim = optimizer.apply_gradients(zip(grads, tvars))  # 更新梯度

    def construct_model(self, lstm_size_each_layer):
        use_basic_cell = self.config.use_basic_cell
        if 0 == use_basic_cell:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(int(size.strip())), output_keep_prob=self.keep_prob) for size in
                                                     lstm_size_each_layer])
        elif 1 == use_basic_cell:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(int(size.strip())), output_keep_prob=self.keep_prob) for size in
                                                     lstm_size_each_layer])
        elif 2 == use_basic_cell:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(int(size.strip())), output_keep_prob=self.keep_prob) for size in
                                                     lstm_size_each_layer])
        return lstm_cell


