# 输入拼接 输入分离 输出拼接 输出分离
import tensorflow as tf
import numpy as np


# 设置参数
n_sample = 2000
# rate = 0.005  # 0.1
epoch = 350
display = 10
batch_size = 512

# 假装有数据
data = np.random.rand(n_sample, 1, 64)
label = np.random.randint(1000, size=n_sample)
input = []
output = []

for i in range(n_sample):
    input.append(np.reshape(data[i], [64]))
    onehot = np.zeros(1000, dtype=float)
    onehot[label[i]] = 1
    output.append(np.reshape(onehot, [1000]))

# 模型
n_input = 64
# n_hidden = 1024
n_layer1 = 200
n_layer2 = 100
n_layer3 = 500
n_layer4 = 300
n_output = 1000

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])


def add_layer(inputs, in_size, out_size, activate_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases

    if activate_function is None:
        outputs = wx_plus_b
    else:
        outputs = activate_function(wx_plus_b)
    return outputs


def model(x):

    layer1 = add_layer(x, n_input, n_layer1, tf.nn.relu)
    layer2 = add_layer(x, n_input, n_layer2, tf.nn.relu)
    layer3_input = tf.concat([layer1, layer2], 1)  # tf.concat

    layer3 = add_layer(layer3_input, n_layer1 + n_layer2, n_layer3, tf.nn.relu)

    layer5_input1, layer4_input = tf.split(layer3, [300, 200], 1)  # tf.split
    layer4 = add_layer(layer4_input, 200, n_layer4, tf.nn.relu)

    layer5_input = tf.concat([layer5_input1, layer4], 1)
    layer5 = add_layer(layer5_input, 300 + n_layer4, n_output)

    return layer5


# 预测
pred = model(x)

# 损失函数和优化策略
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)  # adam不需要设置lr

# 准确率
pred = tf.nn.softmax(pred)  # add softmax here
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(n_sample / batch_size)

    for step in range(epoch):
        acc_total = 0
        co = 0
        for i in range(total_batch):
            _, acc, co = sess.run([optimizer, accuracy, cost],
                                  feed_dict={
                                      x: input[i*batch_size:(i+1)*batch_size],
                                      y: output[i*batch_size:(i+1)*batch_size]})
            acc_total += acc

        if (step + 1) % display == 0:
            print("Step = " + str(step + 1) + ", Accuracy = " + "{:.2f}%".format(100 * acc_total / total_batch))
