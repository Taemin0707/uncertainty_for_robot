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
import time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
from PIL import ImageQt
from PIL import ImageDraw
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
        X_img = tf.reshape(X, [-1, 28, 28, 1])  # img 28x28x1 (black/white)

        # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
        keep_prob = tf.placeholder(tf.float32)

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
        logits = tf.nn.softmax(tf.matmul(L4, W5) + b5)

        with tf.Session() as sess:

            save_path = './models/recognizer_number_cnn_model'
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

            iter = 500

            predict_result = sess.run(logits, feed_dict={X: self.test_input.reshape(1, 784), keep_prob: 0.7})
            number = np.where(predict_result[0] == np.max(predict_result[0]))

            print("-------------------------------------------------------")
            print("number = ", number[0][0])

            # for i in range(iter):
            #     result = sess.run(hypothesis, feed_dict={X: self.test_input.reshape(1, 784), keep_prob: 0.7})
            #     for j in range(10):
            #         number_list[j].append(result[0][j])

            for i in range(iter):
                result = sess.run(logits, feed_dict={X: self.test_input.reshape(1, 784), keep_prob: 0.7})
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
        self.main_image_path = "/home/taemin/images/kist.jpg"
        self.main_image = Image.open(self.main_image_path)
        self.init_widget()
        self.number_recognizer = NumberRecognizer()

    def init_widget(self):
        # 이미지 라벨
        self.main_image_label = QLabel(self)
        pixmap = QPixmap(self.main_image_path)
        self.main_image_label.setPixmap(pixmap)
        self.main_image_label.resize(pixmap.width(), pixmap.height())

        # 버튼
        self.rotate_left = QPushButton("Rotate Left")
        self.new_image = QPushButton("New Image")
        self.rotate_right = QPushButton("Rotate Right")

        self.nope_1 = QPushButton("")
        self.move_up = QPushButton("Move Up")
        self.nope_2 = QPushButton("")

        self.move_left = QPushButton("Move Left")
        self.calculate = QPushButton("Calculate")
        self.move_right = QPushButton("Move Right")

        self.nope_3 = QPushButton("")
        self.move_down = QPushButton("Move Down")
        self.nope_4 = QPushButton("")

        # 버튼 이벤트 처리
        self.rotate_left.clicked.connect(self.rotate_left_button_clicked)
        self.new_image.clicked.connect(self.new_image_button_clicked)
        self.rotate_right.clicked.connect(self.rotate_right_button_clicked)
        self.calculate.clicked.connect(self.calculate_button_clicked)
        self.move_up.clicked.connect(self.move_up_button_clicked)

        # 그래프
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        # 결과창
        self.text = QTextBrowser(self)
        self.text.append("안녕하세요")
        self.text.append("여기에 결과를 보여 주겠습니다.")

        # 레이아웃 설정
        self.left_1_button_layout = QHBoxLayout()
        self.left_1_button_layout.addWidget(self.rotate_left)
        self.left_1_button_layout.addWidget(self.new_image)
        self.left_1_button_layout.addWidget(self.rotate_right)

        self.left_2_button_layout = QHBoxLayout()
        self.left_2_button_layout.addWidget(self.nope_1)
        self.left_2_button_layout.addWidget(self.move_up)
        self.left_2_button_layout.addWidget(self.nope_2)

        self.left_3_button_layout = QHBoxLayout()
        self.left_3_button_layout.addWidget(self.move_left)
        self.left_3_button_layout.addWidget(self.calculate)
        self.left_3_button_layout.addWidget(self.move_right)

        self.left_4_button_layout = QHBoxLayout()
        self.left_4_button_layout.addWidget(self.nope_3)
        self.left_4_button_layout.addWidget(self.move_down)
        self.left_4_button_layout.addWidget(self.nope_4)

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.main_image_label)
        self.left_layout.addLayout(self.left_1_button_layout)
        self.left_layout.addLayout(self.left_2_button_layout)
        self.left_layout.addLayout(self.left_3_button_layout)
        self.left_layout.addLayout(self.left_4_button_layout)

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.text)
        self.right_layout.addWidget(self.canvas)

        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        self.setLayout(self.main_layout)

    def new_image_button_clicked(self):
        fname = QFileDialog.getOpenFileName()
        image_path = fname[0]
        if image_path == None:
            print("Error loading image")
        self.main_image = Image.open(image_path)
        pixmap = QPixmap(image_path)
        self.main_image_path = image_path
        self.main_image_label.setPixmap(pixmap)
        self.main_image_label.resize(pixmap.width(), pixmap.height())
        self.draw_rectangle()

    def rotate_left_button_clicked(self):
        self.main_image = self.main_image.rotate(15)
        new_image = ImageQt.ImageQt(self.main_image)
        new_image = QImage(new_image)
        pixmap = QPixmap.fromImage(new_image)
        self.main_image_label.setPixmap(pixmap)
        self.main_image_label.resize(pixmap.width(), pixmap.height())
        print("회전 완료")

    def rotate_right_button_clicked(self):
        self.main_image = self.main_image.rotate(-15)
        new_image = ImageQt.ImageQt(self.main_image)
        new_image = QImage(new_image)
        pixmap = QPixmap.fromImage(new_image)
        self.main_image_label.setPixmap(pixmap)
        self.main_image_label.resize(pixmap.width(), pixmap.height())
        print("회전 완료")

    def draw_rectangle(self):
        main_image = self.main_image
        width, height = main_image.size
        self.vertex_of_rect = [(width / 2) - 14, (height / 2) - 14, (width / 2) + 14, (height / 2) + 14]
        image_draw = ImageDraw.Draw(main_image)
        image_draw.rectangle(((self.vertex_of_rect[0], self.vertex_of_rect[1]), \
                              (self.vertex_of_rect[2], self.vertex_of_rect[3])), outline='green')
        new_image = ImageQt.ImageQt(main_image)
        new_image = QImage(new_image)
        pixmap = QPixmap.fromImage(new_image)
        self.main_image_label.setPixmap(pixmap)
        self.main_image_label.resize(pixmap.width(), pixmap.height())
        main_image.show()

    def move_up_button_clicked(self):
        self.main_image = Image.open(self.main_image_path)
        self.vertex_of_rect[1] = self.vertex_of_rect[1] - 1
        self.vertex_of_rect[3] = self.vertex_of_rect[3] - 1
        image_draw = ImageDraw.Draw(self.main_image)
        image_draw.rectangle(((self.vertex_of_rect[0], self.vertex_of_rect[1]), \
                              (self.vertex_of_rect[2], self.vertex_of_rect[3])), outline='red')
        new_image = ImageQt.ImageQt(self.main_image)
        new_image = QImage(new_image)
        pixmap = QPixmap.fromImage(new_image)
        self.main_image_label.setPixmap(pixmap)
        self.main_image_label.resize(pixmap.width(), pixmap.height())

    def calculate_button_clicked(self):
        # 시간 측정
        start_time = time.time()
        (mean, variance) = self.number_recognizer.predict_with_dropout(self.main_image)
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

        # 시간 측정 종료
        print("--- %s seconds ---" % (time.time() - start_time))

        # 텍스트 브라우저
        self.text.clear()
        self.text.append("인식 결과는 다음과 같습니다.")
        self.text.append("인식된 숫자는 {} 입니다.".format(first_number))
        self.text.append("후보 3개의 정보는 다음과 같습니다.")
        self.text.append("-------------------------------------")
        self.text.append("1st = {}".format(first_number))
        self.text.append("Mean = {0:.4f}".format(mean_first_number))
        self.text.append("Variance = {0:.4f}".format(var_first_number))
        self.text.append("2nd = {}".format(second_number))
        self.text.append("Mean = {0:.4f}".format(mean_second_number))
        self.text.append("Variance = {0:.4f}".format(var_second_number))
        self.text.append("3rd = {}".format(third_number))
        self.text.append("Mean = {0:.4f}".format(mean_third_number))
        self.text.append("Variance = {0:.4f}".format(var_third_number))
        self.text.append("-------------------------------------")
        self.text.append("계산 소요 시간은 {0:.4f}초 입니다." .format(time.time() - start_time))

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
        ax.set_xlabel('Number')
        ax.set_ylabel('Softmax result')
        ax.set_title('Uncertainty')
        ax.set_xticks(index)
        x_labels = [first_number, second_number, third_number]
        ax.set_xticklabels(x_labels)
        ax.legend()
        self.canvas.draw()
        ax.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = MainWindow()
    example.show()
    sys.exit(app.exec_())