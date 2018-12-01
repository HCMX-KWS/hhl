
import tensorflow as tf
import numpy as np

# 设置参数
n_sample = 10000
# rate = 0.005  # 0.1
epoch = 1000
display = 1
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
n_hidden = 1024
n_output = 1000

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

Weights = {
    'h': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
bias = {
    # bias设置为zeros改变初始点 速度基本不变
    'h': tf.Variable(tf.random_normal([n_hidden])),
    # 'h': tf.Variable(tf.zeros([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
    # 'out': tf.Variable(tf.zeros([n_output]))
}


def model(x, Weights, bias):

    hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, Weights['h']), bias['h']))
    # add hidden_layer2
    out_layer = tf.add(tf.matmul(hidden_layer, Weights['out']), bias['out'])  # remove softmax

    return out_layer


# 预测
pred = model(x, Weights, bias)

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
