import os
import torch
from os.path import exists, dirname
from config.config import classes_path


def getLabels(_classes_path=classes_path):
    assert exists(_classes_path), f'cannot find label_file : {_classes_path}'
    with open(_classes_path, 'r') as f:
        classes = f.readlines()
    return [cls.strip() for cls in classes]


def ensurePath(path, file=True):
    if file:
        path = dirname(path)
    if not exists(path):
        os.makedirs(path)


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def getFuncName(func):
    return f'{func.__module__}.{func.__name__}'


def parseEnumName(enum, length=23):
    fill = (length - len(str(enum))) * ' '
    return f'{enum}{fill}'


