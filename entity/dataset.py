import torch
import random
import numpy as np
from glob import glob
from PIL import Image
from os.path import exists
from config.config import image_size, padValue, multi_scale_train, normalized_labels, align_flip, multi_scale_interval
from assist.image import pad2square, resize, readTensorImage
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


class ImageSet(Dataset):

    def __init__(self, image_folder):
        self.files = sorted(glob("%s/*.*" % image_folder))
        self.count = len(self.files)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        idx = idx % self.count
        image_path = self.files[idx]
        image = Image.open(image_path)
        tensor_image = transforms.ToTensor()(image)
        # 首先填充为方块
        tensor_image, _ = pad2square(tensor_image, padValue)
        # 然后resize
        tensor_image = resize(tensor_image, image_size)
        return image_path, tensor_image


class ListSet(Dataset):

    def __init__(self, file_path):
        print(file_path)
        with open(file_path, 'r') as file:
            image_files = file.readlines()
        image_paths = [path.strip() for path in image_files]
        label_paths = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in
                            image_paths]
        self.image_paths = []
        self.label_paths = []
        for idx, (path, label) in enumerate(zip(image_paths, label_paths)):
            if exists(path) and exists(label):
                self.image_paths.append(path)
                self.label_paths.append(label)
        self.image_size = image_size
        self.min_size = image_size - 3 * 32
        self.max_size = image_size + 3 * 32
        self.batch_count = 0
        self.length = len(self.image_paths)
        self.align_flip = align_flip
        self.normalized_labels = normalized_labels
        self.multi_scale_train = multi_scale_train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        return self.load_data(idx)

    def load_data(self, idx):
        # 信息地址
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = readTensorImage(image_path, transforms.ToTensor())
        # 补充三个纬度
        if len(image) != 3:
            image = image.unsqueeze(0)
            image = image.expand((3, image.shape[1:]))
        _, h, w = image.shape
        # 标记是否为归一化数据
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # 图像填充，统一填充为正方形
        image, pad = pad2square(image, padValue)
        _, padded_h, padded_w = image.shape

        target = None
        # 存在标记文件
        if exists(label_path):
            # 加载标记文件(x, y, w, h)
            boxes = torch.from_numpy(np.loadtxt(label_path)).reshape((-1, 5))
            # 原始标记信息
            origin_center_x, origin_center_y = boxes[:, 1], boxes[:, 2]
            origin_h, origin_w = boxes[:, 3], boxes[:, 4]
            # print(f'origin_target : {boxes}')
            # 转换为四顶点格式
            half_origin_h = origin_h // 2
            half_origin_w = origin_w // 2
            origin_x1 = origin_center_x - half_origin_w
            origin_x2 = origin_center_x + half_origin_w
            origin_y1 = origin_center_y - half_origin_h
            origin_y2 = origin_center_y + half_origin_h
            # 归一化恢复，需要数值进行比较
            adjust_x1, adjust_x2 = w_factor * origin_x1, w_factor * origin_x2
            adjust_y1, adjust_y2 = h_factor * origin_y1, h_factor * origin_y2
            # 换算填充后标记地址
            adjust_x1 += pad[0]
            adjust_x2 += pad[1]
            adjust_y1 += pad[2]
            adjust_y2 += pad[3]
            # 将数据重新归一化(x, y, w, h)
            normalize_x = (adjust_x1 + adjust_x2) / 2 / padded_w
            normalize_y = (adjust_y1 + adjust_y2) / 2 / padded_h
            normalize_h = origin_h * h_factor / padded_h
            normalize_w = origin_w * w_factor / padded_w
            # 回写到box
            boxes[:, 1] = normalize_x
            boxes[:, 2] = normalize_y
            boxes[:, 3] = normalize_w
            boxes[:, 4] = normalize_h
            # 创建容器，填充数据，纬度+1
            target = torch.zeros(len(boxes), 6)
            target[:, 1:] = boxes
            # 额外操作，水平翻转图像
            if self.align_flip and np.random.random() < 0.5:
                # image, target = horizontal_flip(image, target)
                image = torch.flip(image, [-1])
                target[:, 2] = 1 - target[:, 2]
        # print(f'trans_targets : {target}')
        return image_path, image, target

    def collate_fn(self, batch_data):
        image_paths, images, targets = list(zip(*batch_data))
        targets = [target for target in targets if target is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        if self.multi_scale_train and self.batch_count % multi_scale_interval == 0:
            self.image_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        images = torch.stack([resize(image, self.image_size) for image in images])
        self.batch_count += 1
        return image_paths, images, targets
