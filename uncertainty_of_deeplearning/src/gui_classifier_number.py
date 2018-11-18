#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial

This program creates a submenu.

Author: Jan Bodnar
Website: zetcode.com
Last edited: August 2017
"""

import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Classifier Number with MNIST")
        self.setGeometry(10, 10, 640, 480)

        self.init_window()

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

        # 이미지
        self.image_path = "/home/taemin/s-hri_ws/src/uncertainty_for_robot/uncertainty_of_deeplearning/src/images/test.png"
        self.label = QLabel(self)
        pixmap = QPixmap(self.image_path)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(), pixmap.height())

    def show_image(self):
        if self.image_path == None:
            print("Error loading image")

        pixmap = QPixmap(self.image_path)
        self.label.setPixmap(pixmap)
        # self.label.resize(pixmap.width(), pixmap.height())
        self.label.resize(640, 480)

    def import_btn_clicked(self):
        fname = QFileDialog.getOpenFileName()
        self.image_path = fname[0]
        self.show_image()
        print(fname[0])
        # 이미지

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = Example()
    example.show()
    sys.exit(app.exec_())