import tensorflow as tf


class TextCNN_config(object):
    dropout_keep_prob = 0.5
    max_document_length = 600
    num_class = 10
    vocab_size = 5000
    batch_size = 32
    num_epochs = 10
    embedding_size = 64
    filter_sizes = '1, 2, 3, 4, 5'
    num_filters = 128
    learning_rate = 0.001
    l2_reg_lamda = 0
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    # 文本分类 CNN模型
    def __init__(self, config):
        # 作用是让每次生成的都是相同的
        tf.set_random_seed(66)
        # 参数设置
        self.config = config

        # 给输入input_x设置占位符 维度：[None, self.config.max_document_length] None是批次数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_document_length], name='input_x')
        # 同理设置input_y设置占位符 维度：[None, self.config.num_class] None为批次数据
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_class], name='input_y')
        # 设置keep_prob占位符
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # 初始化l2正则初始值
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 设置总词表[self.config.vocab_size, self.config.embedding_size]，取值范围[-1.0,1.0]
            self.W = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.embedding_size], -1.0, 1.0)
                , name='W')
            # 根据词表查找对应的词向量
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 在后面添加一个维度变成4维方便后续卷积
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        # 根据不同的卷积核大小进行卷积操作 for循环
        for i, filters_size in enumerate(list(map(int, self.config.filter_sizes.split(",")))):
            with tf.name_scope("conv-maxpool-%s"%filters_size):
                # 卷积层
                # 卷积核，高度，宽度，通道数，卷积核个数
                filters_shape = [filters_size, self.config.embedding_size, 1, self.config.num_filters]
                # 根据filters_shape初始化W值
                W = tf.Variable(tf.truncated_normal(filters_shape, stddev=0.1), name='W')
                # 根据卷积核数目初始化b，卷积核数目就是所要提取的特征数目
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name='b')
                # 卷积操作
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                # 加上偏置值并进行relu操作
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # 池化层
                pooled = tf.nn.max_pool(
                    h,
                    # 卷积后的维度
                    ksize=[1, self.config.max_document_length-filters_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool"
                )
                # 将几种卷积核尺寸卷积后的添加到pooled_outputs
                pooled_outputs.append(pooled)

        # combine all the pooled features
        # 总计特征维度
        num_filters_total = self.config.num_filters*len(self.config.filter_sizes.split(","))
        # 将第三个维度进行累加
        self.h_pool = tf.concat(pooled_outputs, 3)
        # -1表示根据批次数自动计算维度
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 定义attention layer
        attention_size = num_filters_total
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([num_filters_total, attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            # 取出的outputs[t]是每个单词输入后得到的结果，[batch_size,2*rnn_size(双向)]
            # 因此u_t可以认为是对这个单词结果的打分,[batch_size,attention_size]
            u_t = tf.tanh(tf.matmul(self.h_pool_flat, attention_w) + attention_b)  # x * num_filters_total
            # 最终得到的u_list是[sequence_length,batch_size,attention_size],是每一个单词的权重
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            # 将权重变为[batch_size,1]
            z_t = tf.matmul(u_t, u_w)  # x * 1
            # 取概率权重
            print(z_t.shape)
            self.alpha = tf.nn.softmax(z_t)
            # [batch_size,num_filters_total]*[batch_size,1]=[batch_size,num_filters_total],实际就是对每一个乘以一个权重
            self.final_output = self.h_pool_flat * self.alpha  # x * num_filters_total

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.final_output, self.keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, self.config.num_class],
                    # 初始化权重矩阵
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_class]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
                self.scores = tf.nn.softmax(self.logits)
                self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            # 确定惩罚力度
            self.loss = tf.reduce_mean(losses)+self.config.l2_reg_lamda*l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # 转换为浮点型的准确率
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        with tf.name_scope("optim"):
            # 优化函数
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)





















