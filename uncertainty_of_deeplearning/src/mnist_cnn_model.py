#-*-coding:utf-8-*-

import os
import sys
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

class NumberRecognizer(object):
    """숫자 인식 클래스.
    위 숫자 인식 클래스의 기능으로는
    1) 학습용 숫자 이미지 전처리
    2)

    """
    def __init__(self):
        print(sys.version)
        # for reproducibility
        tf.set_random_seed(777)
        self.TRAIN_DIR = '/home/taemin/MNIST/trainingSet'
        self.TEST_DIR = '/home/taemin/MNIST/testSet'
        self.train_input = []
        self.train_label = []
        self.test_input = []
        self.test_label = []

    def preprocess_image(self):
        # 학습용 데이터 처리
        train_folder_list = np.array(os.listdir(self.TRAIN_DIR))
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(train_folder_list)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        train_input = []
        train_label = []

        for index in range(len(train_folder_list)):
            path = os.path.join(self.TRAIN_DIR, train_folder_list[index])
            path = path + '/'
            image_list = os.listdir(path)

            for image in image_list:
                image_path = os.path.join(path, image)
                image = Image.open(image_path)
                train_input.append(np.array(image))
                train_label.append(np.array(onehot_encoded[index]))

        train_input = np.reshape(train_input, (-1, 784))
        train_label = np.reshape(train_label, (-1, 10))
        train_input = np.array(train_input).astype(np.float32)
        train_label = np.array(train_label).astype(np.float32)
        # np.save('train_data.npy', train_input)
        # np.save('train_label.npy', train_label)

        s_train = np.arange(train_input.shape[0])
        np.random.shuffle(s_train)
        train_input = train_input[s_train]
        train_label = train_label[s_train]
        self.train_input = train_input
        self.train_label = train_label

        # 검증용 데이터 처리
        test_folder_list = np.array(os.listdir(self.TEST_DIR))
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(test_folder_list)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        test_input = []
        test_label = []

        for index in range(len(test_folder_list)):
            path = os.path.join(self.TEST_DIR, test_folder_list[index])
            path = path + '/'
            image_list = os.listdir(path)

            for image in image_list:
                image_path = os.path.join(path, image)
                image = Image.open(image_path)
                test_input.append(np.array(image))
                test_label.append(np.array(onehot_encoded[index]))

        test_input = np.reshape(test_input, (-1, 784))
        test_label = np.reshape(test_label, (-1, 10))
        test_input = np.array(test_input).astype(np.float32)
        test_label = np.array(test_label).astype(np.float32)
        # np.save('test_data.npy', test_input)
        # np.save('test_label.npy', test_label)

        s_test = np.arange(test_input.shape[0])
        np.random.shuffle(s_test)
        test_input = test_input[s_test]
        test_label = test_label[s_test]
        self.test_input = test_input
        self.test_label = test_label

        print(self.train_input.shape)
        print(self.train_label.shape)
        print(self.test_input.shape)
        print(self.test_label.shape)

    def network_model(self):
        # parameters
        learning_rate = 0.001
        training_epochs = 15
        batch_size = 100

        # input place holders
        X = tf.placeholder(tf.float32, [None, 784])
        X_img = tf.reshape(X, [-1, 28, 28, 1])  # img 28x28x1 (black/white)
        Y = tf.placeholder(tf.float32, [None, 10])

        # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
        keep_prob = tf.placeholder(tf.float32)

        # L1 ImgIn shape=(?, 28, 28, 1)
        W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
        '''
        Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
        Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
        Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
        Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
        '''

        # L2 ImgIn shape=(?, 14, 14, 32)
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
        '''
        Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
        Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
        Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
        Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
        '''

        # L3 ImgIn shape=(?, 7, 7, 64)
        W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='SAME')
        L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
        L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
        '''
        Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
        Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
        Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
        Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
        Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
        '''

        # L4 FC 4x4x128 inputs -> 625 outputs
        W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                             initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([625]))
        L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
        L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
        '''
        Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
        Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
        '''

        # L5 Final FC 625 inputs -> 10 outputs
        W5 = tf.get_variable("W5", shape=[625, 10],
                             initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([10]))
        logits = tf.matmul(L4, W5) + b5

        # define cost/loss & optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # initialize
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # train my model
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(self.train_input) / batch_size)

            for i in range(total_batch):
                start = ((i+1) * batch_size) - batch_size
                end = ((i+1) * batch_size)
                batch_xs = self.train_input[start:end]
                batch_ys = self.train_label[start:end]
                feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
                c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Learning Finished!')

        # Test model and check accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
            X: self.test_input, Y: self.test_label, keep_prob: 1}))

        # Save the model
        save_path = './models/recognizer_number_cnn_model'
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print("Model saved in file: %s" %save_path)
        sess.close()

if __name__ == "__main__":
    number_recognizer = NumberRecognizer()
    number_recognizer.preprocess_image()
    number_recognizer.network_model()