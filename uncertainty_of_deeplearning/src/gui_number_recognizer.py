#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This program creates a Number-Recognizer.

Author: Taemin Choi
Email: choitm0707@kist.re.kr
Last edited: November 2018
"""

import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
from PIL import ImageQt
import numpy as np
import tensorflow as tf

class NumberRecognizer(object):
    """

    """
    def __init__(self):
        # print(sys.version)
        # for reproducibility
        tf.set_random_seed(777)

    def predict_with_dropout(self, image):
        # 예측용 이미지 데이터 처리
        self.test_input = []
        test_image = image
        self.test_input.append(np.array(test_image))
        self.test_input = np.reshape(self.test_input, (-1, 784))
        print("테스트 이미지 = ", self.test_input.shape)

        tf.reset_default_graph()

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

            predict_result = sess.run(hypothesis, feed_dict={X: self.test_input.reshape(1, 784), keep_prob: 0.7})
            number = np.where(predict_result[0] == np.max(predict_result[0]))

            print("-------------------------------------------------------")
            print("number = ", number[0][0])

            # for i in range(iter):
            #     result = sess.run(hypothesis, feed_dict={X: self.test_input.reshape(1, 784), keep_prob: 0.7})
            #     for j in range(10):
            #         number_list[j].append(result[0][j])

            for i in range(iter):
                result = sess.run(hypothesis, feed_dict={X: self.test_input.reshape(1, 784), keep_prob: 0.7})
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
            return mean, variance

class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Number-Recognizer")
        self.setGeometry(10, 10, 1280, 960)

        self.init_window()
        self.form_widget = FormWidget(self)
        self.setCentralWidget(self.form_widget)

    def init_window(self):
        # 메뉴바
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        importMenu = menubar.addMenu('Import')

        # 서브 메뉴들
        newAct = QAction('New', self)
        newAct.triggered.connect(self.btn1_clicked)
        fileMenu.addAction(newAct)

        impAct = QAction('Import image', self)
        impAct.triggered.connect(self.import_btn_clicked)
        fileMenu.addAction(impAct)

        saveAct = QAction('Save result', self)
        saveAct.triggered.connect(self.btn1_clicked)
        fileMenu.addAction(saveAct)

        closeAct = QAction('Close', self)
        closeAct.triggered.connect(self.btn1_clicked)
        fileMenu.addAction(closeAct)

    def import_btn_clicked(self):
        fname = QFileDialog.getOpenFileName()
        self.form_widget.show_image(fname[0])

    def btn1_clicked(self):
        QMessageBox.about(self, "message", "clicked")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

class FormWidget(QWidget):

    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        self.image_path = "/home/taemin/images/kist.jpg"
        self.image_array = []
        self.image = Image.open(self.image_path)
        self.init_widget()
        self.number_recognizer = NumberRecognizer()

    def init_widget(self):
        # 이미지 라벨
        self.label = QLabel(self)
        pixmap = QPixmap(self.image_path)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(), pixmap.height())

        # 버튼
        self.rotate_left = QPushButton("Rotate Left")
        self.new_image = QPushButton("New Image")
        self.rotate_right = QPushButton("Rotate Right")
        self.calculate = QPushButton("Calculate")

        # 버튼 이벤트 처리
        self.rotate_left.clicked.connect(self.rotate_left_button_clicked)
        self.new_image.clicked.connect(self.new_image_button_clicked)
        self.rotate_right.clicked.connect(self.rotate_right_button_clicked)
        self.calculate.clicked.connect(self.calculate_button_clicked)

        # 그래프
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        # 결과창
        self.text = QTextBrowser(self)
        self.text.append("안녕하세요")
        self.text.append("여기에 결과를 보여 주겠습니다.")

        # 레이아웃 설정
        self.left_button_layout = QHBoxLayout()
        self.left_button_layout.addWidget(self.rotate_left)
        self.left_button_layout.addWidget(self.new_image)
        self.left_button_layout.addWidget(self.rotate_right)

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.label)
        self.left_layout.addLayout(self.left_button_layout)
        self.left_layout.addWidget(self.calculate)

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.text)
        self.right_layout.addWidget(self.canvas)

        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        self.setLayout(self.main_layout)

    def show_image(self, image_path_):
        self.image_path = image_path_
        if self.image_path == None:
            print("Error loading image")
        pixmap = QPixmap(self.image_path)
        print("pixmap = ", pixmap)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(), pixmap.height())

    def new_image_button_clicked(self):
        fname = QFileDialog.getOpenFileName()
        self.image_path = fname[0]
        self.show_image(self.image_path)
        self.image = Image.open(self.image_path)
        self.image_array.clear()
        self.image_array.append(np.array(self.image))
        # print(self.image_array)

    def rotate_left_button_clicked(self):
        self.image = self.image.rotate(15)
        new_image = ImageQt.ImageQt(self.image)
        new_image = QImage(new_image)
        pixmap = QPixmap.fromImage(new_image)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(), pixmap.height())
        print("회전 완료")

    def rotate_right_button_clicked(self):
        self.image = self.image.rotate(-15)
        new_image = ImageQt.ImageQt(self.image)
        new_image = QImage(new_image)
        pixmap = QPixmap.fromImage(new_image)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(), pixmap.height())
        print("회전 완료")

    def calculate_button_clicked(self):
        (mean, variance) = self.number_recognizer.predict_with_dropout(self.image)
        sorted_mean = np.sort(mean)
        # print(sorted_mean)
        first_number = np.where(mean == sorted_mean[9])
        first_number = first_number[0][0]
        mean_first_number = mean[first_number]
        var_first_number = variance[first_number]
        second_number = np.where(mean == sorted_mean[8])
        second_number = second_number[0][0]
        mean_second_number = mean[second_number]
        var_second_number = variance[second_number]
        third_number = np.where(mean == sorted_mean[7])
        third_number = third_number[0][0]
        mean_third_number = mean[third_number]
        var_third_number = variance[third_number]

        # 텍스트 브라우저
        self.text.clear()
        self.text.append("인식 결과는 다음과 같습니다.")
        self.text.append("인식된 숫자는 = {}".format(first_number))
        self.text.append("후보 3개는 다음과 같습니다.")
        self.text.append("1st = {}".format(first_number))
        self.text.append("2nd = {}".format(second_number))
        self.text.append("3rd = {}".format(third_number))

        # 그래프 관련
        n_groups = 3
        means = (mean_first_number, mean_second_number, mean_third_number)
        variances = (var_first_number, var_second_number, var_third_number)

        ax = self.fig.add_subplot(1, 1, 1)
        index = np.arange(n_groups)

        # 막대 사이의 거리
        bar_width = 0.3

        # 막대 그래프
        rect1 = ax.bar(0, mean_first_number, bar_width, yerr=var_first_number, capsize=3, ecolor='r', label='First')
        rect2 = ax.bar(1, mean_second_number, bar_width, yerr=var_second_number, capsize=3, ecolor='r', label='Second')
        rect3 = ax.bar(2, mean_third_number, bar_width, yerr=var_third_number, capsize=3, ecolor='r', label='Third')
        # rect = ax.bar(index, means, bar_width, yerr=variances, capsize=3, ecolor='r', label='Top3 of Numbers')
        ax.set_xlabel('Number')
        ax.set_ylabel('Softmax result')
        ax.set_title('Uncertainty')
        # ax.set_xticks(index, ('{}'.format(first_number), '{}'.format(second_number), '{}'.format(third_number)))
        ax.set_xticks(index)
        x_labels = [first_number, second_number, third_number]
        ax.set_xticklabels(x_labels)
        ax.legend()
        self.canvas.draw()
        ax.clear()
        # ax = self.fig.add_subplot(1, 1, 1)
        # graph_x1 = [1]
        # graph_x2 = [2]
        # graph_x3 = [3]
        # graph_y1 = [1]
        # graph_y2 = [2]
        # graph_y3 = [3]
        # ax.plot(graph_x1, graph_y1, 'C0', lw=2)
        # ax.plot(graph_x2, graph_y2, 'C0', lw=2)
        # ax.plot(graph_x3, graph_y3, 'C0', lw=2)
        # self.canvas.draw()

        # print(type(image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = MainWindow()
    example.show()
    sys.exit(app.exec_())