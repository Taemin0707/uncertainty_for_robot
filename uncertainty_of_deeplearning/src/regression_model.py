#-*-coding:utf-8-*-

import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

class DeepLearning(object):
    """일반적인 딥러닝 모델
    """

    def __init__(self):
        print("Vesion Check!")
        print("python {}".format(sys.version))
        print("tensorflow version {}".format(tf.VERSION))

        tf.set_random_seed(777)

        self.delta = 0.001
        self.nub_node = 512

    def network_model(self):
        # 학습을 위한 데이터셋을 정의한다.
        delta = self.delta
        x_train_data = np.arange(-1, 0.5, delta)
        x_datas = x_train_data.flatten()
        function = [2 * math.sin(x) + x + 1 for x in x_datas]
        y_train_data = np.array(function).flatten()

        x_train_data = x_train_data.reshape(len(x_train_data), 1)
        y_train_data = y_train_data.reshape(len(y_train_data), 1)

        x_data = np.array(x_train_data, dtype=np.float32)
        y_data = np.array(y_train_data, dtype=np.float32)

        # parameters
        learning_rate = 0.001
        training_epochs = 10000
        nub_node = self.nub_node

        X = tf.placeholder(tf.float32, [None, 1])
        Y = tf.placeholder(tf.float32, [None, 1])

        # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
        keep_prob = tf.placeholder(tf.float32)

        # weights & bias for nn layers
        W1 = tf.get_variable("W1", shape=[1, nub_node], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([nub_node]))
        L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        D1 = tf.nn.dropout(L1, keep_prob=keep_prob)

        W2 = tf.get_variable("W2", shape=[nub_node, nub_node], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([nub_node]))
        L2 = tf.nn.relu(tf.matmul(D1, W2) + b2)
        D2 = tf.nn.dropout(L2, keep_prob=keep_prob)

        W3 = tf.get_variable("W3", shape=[nub_node, nub_node], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.Variable(tf.random_normal([nub_node]))
        L3 = tf.nn.relu(tf.matmul(D2, W3) + b3)
        D3 = tf.nn.dropout(L3, keep_prob=keep_prob)

        # W4 = tf.get_variable("W4", shape=[nub_node, nub_node], initializer=tf.contrib.layers.xavier_initializer())
        # b4 = tf.Variable(tf.random_normal([nub_node]))
        # L4 = tf.nn.relu(tf.matmul(D3, W4) + b4)
        # D4 = tf.nn.dropout(L4, keep_prob=keep_prob)

        W5 = tf.get_variable("W5", shape=[nub_node, 1], initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([1]))
        hypothesis = tf.matmul(D3, W5) + b5

        # cost/loss function
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        # Minimize
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        # Launch the graph in a session.
        sess = tf.Session()
        # Initializes global variables in the graph.
        sess.run(tf.global_variables_initializer())

        # Fit the line
        for step in range(training_epochs):
            cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
            if step % 1000 == 0:
                print(step, cost_val)

        print('Learning Finished!')

        # Save the model
        save_path = './models/linear_regression_model'
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print("Model saved in file: %s" %save_path)
        sess.close()

    def predict_with_dropout(self):
        delta = self.delta
        self.test_data = np.arange(-1, 1, delta).reshape(-1, 1)
        # print(len(self.test_data))
        # print("테스트 입력값 = ", self.test_data)

        tf.reset_default_graph()

        # input place holders
        X = tf.placeholder(tf.float32, [None, 1])

        # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
        keep_prob = tf.placeholder(tf.float32)

        nub_node = self.nub_node

        # weights & bias for nn layers
        W1 = tf.get_variable("W1", shape=[1, nub_node], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([nub_node]))
        L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        D1 = tf.nn.dropout(L1, keep_prob=keep_prob)

        W2 = tf.get_variable("W2", shape=[nub_node, nub_node], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([nub_node]))
        L2 = tf.nn.relu(tf.matmul(D1, W2) + b2)
        D2 = tf.nn.dropout(L2, keep_prob=keep_prob)

        W3 = tf.get_variable("W3", shape=[nub_node, nub_node], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.Variable(tf.random_normal([nub_node]))
        L3 = tf.nn.relu(tf.matmul(D2, W3) + b3)
        D3 = tf.nn.dropout(L3, keep_prob=keep_prob)

        W4 = tf.get_variable("W4", shape=[nub_node, nub_node], initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([nub_node]))
        L4 = tf.nn.relu(tf.matmul(D3, W4) + b4)
        D4 = tf.nn.dropout(L4, keep_prob=keep_prob)

        W5 = tf.get_variable("W5", shape=[nub_node, 1], initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([1]))
        hypothesis = tf.matmul(D4, W5) + b5

        with tf.Session() as sess:
            save_path = './models/linear_regression_model'
            new_saver = tf.train.Saver()
            new_saver.restore(sess, save_path)

            result = []
            iter = 1

            for i in range(iter):
                result.append(sess.run(hypothesis, feed_dict={X: self.test_data, keep_prob: 1}))
            result = np.array(result).reshape(iter, len(self.test_data)).T

        return result

class Visualization(object):
    def __init__(self):
        self.ylim = (-10, 10)
        self.fig = plt.figure(figsize=(18, 5))
        self.fig.subplots_adjust(hspace=0.13, wspace=0.05)
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

    def vertical_line_trainin_range(self, x_train_):
        minx, maxx = np.min(x_train_), np.max(x_train_)
        self.ax1.axvline(maxx, c="red", ls="--")
        self.ax1.axvline(minx, c="red", ls="--", label="The range of the x_train")
        self.ax2.axvline(maxx, c="red", ls="--")
        self.ax2.axvline(minx, c="red", ls="--", label="The range of the x_train")

    def plot_y_pred(self, x_test_, y_prediction_):
        self.ax1.plot(x_test_, y_prediction_, color="yellow", label="y_pred")

    def plot_y_train(self, x_train_, y_train_):
        self.ax1.plot(x_train_, y_train_, color="blue", label="y_train")
        # self.ax1.set_xlabel("x")

    def plot_y_pred_with_dropout(self, x_test_, y_prediction_with_dropout_):

        for iiter in range(y_prediction_with_dropout_.shape[1]):
            self.ax2.plot(x_test_, y_prediction_with_dropout_[:, iiter], alpha=0.05)
        self.ax2.set_title("Predicitions with Dropout", fontsize="10")

    def finish(self):
        box1 = self.ax1.get_position()
        self.ax1.set_position([box1.x0, box1.y0 + box1.height * 0.1, box1.width, box1.height * 0.9])
        box2 = self.ax2.get_position()
        self.ax2.set_position([box2.x0, box2.y0 + box2.height * 0.1, box2.width, box2.height * 0.9])

        self.ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), \
        fancybox=True, shadow=True, ncol=5)
        self.ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), \
        fancybox=True, shadow=True, ncol=5)

        self.ax1.set_ylim(self.ylim)
        self.ax2.set_ylim(self.ylim)

        self.ax1.set_xlabel("x")
        self.ax2.set_xlabel("x")
        plt.show()


if __name__ == '__main__':
    # 노드를 초기화 한다.
    # rospy.init_node('db_management', anonymous=True)

    # 학습을 위한 데이터셋을 정의한다.
    delta = 0.001
    x_train_data = np.arange(-1, 0.5, delta)
    x_datas = x_train_data.flatten()
    function = [2 * math.sin(x * (2 * math.pi)) + x + 1 for x in x_datas]
    y_train_data = np.array(function).flatten()
    x_test_data = np.arange(-1, 1, delta).reshape(-1, 1)

    x_train_data = x_train_data.reshape(len(x_train_data), 1)
    y_train_data = y_train_data.reshape(len(y_train_data), 1)
    print("x_train.shape={}".format(x_train_data.shape))
    print("y_train.shape={}".format(y_train_data.shape))
    print("x_train -- Min:{:4.3f} Max:{:4.3f}".format(np.min(x_train_data), np.max(x_train_data)))
    print("x_test  -- Min:{:4.3f} Max:{:4.3f}".format(np.min(x_test_data), np.max(x_test_data)))

    learning = DeepLearning()
    learning.network_model()
    # visualization = Visualization()
    # y_prediction_with_dropout = learning.predict_with_dropout()

    # y_prediction = model.predict(x_test_data)


    # visualization.vertical_line_trainin_range(x_train_data)
    # visualization.plot_y_pred(x_test_data, y_prediction)

    # visualization.plot_y_train(x_train_data, y_train_data)
    # visualization.plot_y_pred_with_dropout(x_test_data, y_prediction_with_dropout)
    # visualization.finish()
    # print("Start Guassian")
    # gaussian_process = GaussianProcess(x_test_data, y_prediction_with_dropout)
    # gaussian_process.predict_with_gp()


    # rospy.spin()
