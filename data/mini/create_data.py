import numpy as np
import os
from os.path import exists
from config.config import mini_limit


def mini_filter(data_path, limit):
    with open(data_path, 'r') as f:
        images_path = [line.strip() for line in f.readlines()]
    labels_path = [path.replace('images', 'labels')
                       .replace('png', 'txt')
                       .replace('jpg', 'txt') for path in images_path]
    pair_images = []
    pair_labels = []
    for (image_path, label_path) in zip(images_path, labels_path):
        if exists(image_path) and exists(label_path):
            pair_images.append(image_path)
            pair_labels.append(label_path)

    data_count = len(pair_images)
    count = data_count if data_count < limit else limit
    return np.random.choice(np.array(pair_images), count, replace=False), np.random.choice(np.array(pair_labels), count,
                                                                                           replace=False)


def mini_save(lines, name):
    with open(name, 'w') as f:
        f.writelines([f'{line}\n' for line in lines])


def mini_create(read_path, save_path):
    images_data, labels_data = mini_filter(read_path, limit=mini_limit)
    mini_save(images_data, save_path)
    detect_save(images_data)


def detect_save(names):
    for name in names:
        os.system(f'cp {name} ../detect/')


if __name__ == '__main__':
    mini_create('../coco/train.txt', 'train.txt')
    mini_create('../coco/valid.txt', 'valid.txt')
