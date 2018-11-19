#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial

This program creates a submenu.

Author: Jan Bodnar
Website: zetcode.com
Last edited: August 2017
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
        pass
        # print(type(image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = MainWindow()
    example.show()
    sys.exit(app.exec_())