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

    def predict_with_dropout(self, image_path):
        # target number
        image = Image.open('./')
        test_number = 4000
        test_image = self.train_input[test_number]
        test_image = test_image.reshape(28, 28)
        test_image = Image.fromarray(test_image)
        test_image.show()

        # input place holders
        X = tf.placeholder(tf.float32, [None, 784])

        # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
        keep_prob = tf.placeholder(tf.float32)

        # weights & bias for nn layers
        W1 = tf.get_variable("W1", shape=[784, 512],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([512]))
        L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

        W2 = tf.get_variable("W2", shape=[512, 512],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([512]))
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

        W3 = tf.get_variable("W3", shape=[512, 512],
                             initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.Variable(tf.random_normal([512]))
        L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
        L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

        W4 = tf.get_variable("W4", shape=[512, 512],
                             initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([512]))
        L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
        L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

        W5 = tf.get_variable("W5", shape=[512, 10],
                             initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([10]))
        hypothesis = tf.nn.softmax(tf.matmul(L4, W5) + b5)

        # target input
        test_number = 4000
        test_image = self.train_input[test_number]
        test_image = test_image.reshape(28, 28)
        test_image = Image.fromarray(test_image)
        test_image.show()

        with tf.Session() as sess:
            save_path = './models/classifier_number_model'
            new_saver = tf.train.Saver()
            new_saver.restore(sess, save_path)

            number_list = []
            number_zero = []
            number_one = []
            number_two = []
            number_three = []
            number_four = []
            number_five = []
            number_six = []
            number_seven = []
            number_eight = []
            number_nine = []

            iter = 100

            predict_result = sess.run(hypothesis, feed_dict={X: self.train_input[test_number].reshape(1, 784), keep_prob: 0.7})
            number = np.where(predict_result[0] == np.max(predict_result[0]))

            print("-------------------------------------------------------")
            print("number = ", number)

            for i in range(iter):
                result = sess.run(hypothesis, feed_dict={X: self.train_input[test_number].reshape(1, 784), keep_prob: 0.7})
                for j in range(10):
                    number_list[j].append(result[0][j])

            for i in range(iter):
                result = sess.run(hypothesis, feed_dict={X: self.train_input[test_number].reshape(1, 784), keep_prob: 0.7})
                # print(result.shape)
                number_zero.append(result[0][0])
                number_one.append(result[0][1])
                number_two.append(result[0][2])
                number_three.append(result[0][3])
                number_four.append(result[0][4])
                number_five.append(result[0][5])
                number_six.append(result[0][6])
                number_seven.append(result[0][7])
                number_eight.append(result[0][8])
                number_nine.append(result[0][9])

            variance_zero = np.var(number_zero)
            variance_one = np.var(number_one)
            variance_two = np.var(number_two)
            variance_three = np.var(number_three)
            variance_four = np.var(number_four)
            variance_five = np.var(number_five)
            variance_six = np.var(number_six)
            variance_seven = np.var(number_seven)
            variance_eight = np.var(number_eight)
            variance_nine = np.var(number_nine)

            mean_zero = np.mean(number_zero)
            mean_one = np.mean(number_one)
            mean_two = np.mean(number_two)
            mean_three = np.mean(number_three)
            mean_four = np.mean(number_four)
            mean_five = np.mean(number_five)
            mean_six = np.mean(number_six)
            mean_seven = np.mean(number_seven)
            mean_eight = np.mean(number_eight)
            mean_nine = np.mean(number_nine)

            variance = [variance_zero, variance_one, variance_two, variance_three, variance_four,\
                        variance_five, variance_six, variance_seven, variance_eight, variance_nine]

            mean = [mean_zero, mean_one, mean_two, mean_three, mean_four, \
                    mean_five, mean_six, mean_seven, mean_eight, mean_nine]

            print("mean = ", mean)
            print("variance = ", variance)
            # index = np.where(variance == np.min(variance))
            # print("result = ", index)

if __name__ == "__main__":
    number_recognizer = NumberRecognizer()
    number_recognizer.preprocess_image()
    number_recognizer.predict_with_dropout()