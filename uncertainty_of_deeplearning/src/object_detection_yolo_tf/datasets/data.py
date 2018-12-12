#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This program creates a YOLO model with Tensorflow.

Author: Taemin Choi
Email: choitm0707@kist.re.kr
Last edited: December 2018
"""

import os
import json
import glob
import numpy as np
from PIL import Image

IM_EXTENSIONS = ['png', 'jpg', 'bmp']

def read_data():
    """
    이미지 데이터를 로드하고, 전처리 수행
    :param data_dir: (str) 데이터가 저장된 경로
    :param image_size: (tuple)이미지 리사이즈 크기
    :param pixels_per_grid: (int) 한 그리드 당 실제 사이즈
    :param no_label: (bool) 레이블을 로드할 지 여부
    :return: X_set: np.ndarray, shape:(N, H, W, C).
           : Y_set: np.ndarray, shape:(N, g_H, g_W, ANCHORS, 5 + num_classes).
    """

    # for test
    data_dir = '/home/taemin/s-hri_ws/src/uncertainty_for_robot/uncertainty_of_deeplearning/src/object_detection_yolo_tf/face/train'
    image_size = [416, 416]
    pixels_per_grid = 32
    no_label = False

    image_dir = os.path.join(data_dir, 'images')
    class_map_path = os.path.join(data_dir, 'classes.json')
    anchors_path = os.path.join(data_dir, 'anchors.json')
    class_map = load_json(class_map_path)
    anchors = load_json(anchors_path)
    num_classes = len(class_map)
    grid_h, grid_w = [image_size[i] // pixels_per_grid for i in range(2)]
    image_paths = []
    for extension in IM_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(image_dir, '*.{}'.format(extension))))
    anno_dir = os.path.join(data_dir, 'annotations')
    images = []
    labels = []

    image_path = image_paths[0]
    # 이미지를 읽어오고 사이즈를 변환한다
    raw_image = Image.open(image_path)
    resized_image = raw_image.resize((image_size[0], image_size[1]))
    raw_image = np.array(raw_image, dtype=np.float32)
    image_original_size = raw_image.shape[:2]
    resized_image = np.array(resized_image, dtype=np.float32)
    if len(resized_image.shape) == 2:
        # image shape = (w, h, 3)
        # 만약 RGB 채널이 없다면, 끝에 한 차원 늘리고 배열을 3번 중첩하여 (w, h, 3) 로 만든다
        resized_image = np.expand_dims(resized_image, 2)
        resized_image = np.concatenate([resized_image, resized_image, resized_image], -1)
    images.append(resized_image)

    # # 라벨이 없다면
    # if no_label:
    #     labels.append(0)
    #     # print("hello")
    #     continue
    print("what")
    # 바운딩 박스를 로드하고 YOLO 모델에 맞게 변환한다
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    anno_path = os.path.join(anno_dir, '{}.anno'.format(image_name))
    anno = load_json(anno_path)
    print("grid_h = ", grid_h)
    print("grid_w = ", grid_w)
    print("length of anchors = ", len(anchors))
    print("anchors = ", anchors)
    print("number of classes = ", num_classes)
    print("anno = ", anno)
    # label = np.zeros((grid_h, grid_w, len(anchors), 5 + num_classes))
    # print(label)


def load_json(json_path):
    """
    json 파일 읽어오기
    :param json_path: json 파일 경로
    :return: json 파일 데이터
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    # WORK_SPACE = '/home/taemin/s-hri_ws/src/uncertainty_for_robot/uncertainty_of_deeplearning/src/object_detection_yolo_tf/face/train'

    read_data()