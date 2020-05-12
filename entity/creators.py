import numpy as np
from torch import nn
from net import EmptyLayer, YOLOLayer
from entity.enum import LayerType
from entity.parser import LayerDefinition
from registry.creatorRegistry import creator


# 卷积层的生成器
@creator(LayerType.CONVOLUTIONAL)
def ConvolutionalCreator(definition: LayerDefinition, order: int, filterCollector: list, lastFilter):
    bn = definition.intValue('batch_normalize')
    currentFilter = definition.intValue('filters')
    kernel_size = definition.intValue('size')
    stride = definition.intValue('stride')
    padding = (kernel_size - 1) // 2
    model = nn.Sequential()
    model.add_module(
        f'conv_{order}',
        nn.Conv2d(
            in_channels=filterCollector[-1],
            out_channels=currentFilter,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=not bn
        )
    )
    if bn:
        model.add_module(
            f'bn_{order}',
            nn.BatchNorm2d(currentFilter, momentum=0.9, eps=1e-5)
        )
    if 'leaky' == definition.stringValue('activation'):
        model.add_module(
            f'leaky_{order}',
            nn.LeakyReLU(0.1)
        )
    return currentFilter, model


# 池化层的生成器
@creator(LayerType.MAXPOOLING)
def MaxPoolingCreator(definition: LayerDefinition, order: int, filterCollector: list, lastFilter):
    kernel_size = definition.intValue('size')
    stride = definition.intValue('stride')
    padding = (kernel_size - 1) // 2
    model = nn.Sequential()
    if kernel_size == 2 and stride == 1:
        model.add_module(
            f'_debug_padding_{order}',
            nn.ZeroPad2d((0, 1, 0, 1))
        )
    model.add_module(
        f'maxpooling_{order}',
        nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    )
    return lastFilter, model


# 上采样的生成器
@creator(LayerType.UPSAMPLE)
def UpSampleCreator(definition: LayerDefinition, order: int, filterCollector: list, lastFilter):
    stride = definition.intValue('stride')
    model = nn.Sequential()
    model.add_module(
        f'upsample_{order}',
        nn.Upsample(scale_factor=stride, mode='nearest')
    )
    return lastFilter, model


# 路由层的生成器
@creator(LayerType.ROUTE)
def RouteCreator(definition: LayerDefinition, order: int, filterCollector: list, lastFilter):
    layers = [int(x.strip()) for x in definition.stringValue('layers').split(',')]
    currentFilter = sum([filterCollector[1:][i] for i in layers])
    model = nn.Sequential()
    model.add_module(
        f'route_{order}',
        EmptyLayer()
    )
    return currentFilter, model


# 快照层生成器
@creator(LayerType.SHORTCUT)
def ShortcutCreator(definition: LayerDefinition, order: int, filterCollector: list, lastFilter):
    currentFilter = filterCollector[1:][definition.intValue('from')]
    model = nn.Sequential()
    model.add_module(
        f'shortcut_{order}',
        EmptyLayer()
    )
    return currentFilter, model


# YOLO层生成器
@creator(LayerType.YOLO)
def YOLOCreator(definition: LayerDefinition, order: int, filterCollector: list, lastFilter):
    anchor_idx = [int(x.strip()) for x in definition.stringValue('mask').split(',')]
    anchors = [int(x.strip()) for x in definition.stringValue('anchors').split(',')]
    anchors = np.array(anchors).reshape(-1, 2)[anchor_idx].tolist()
    classes = definition.intValue('classes')
    image_size = definition.intValue('height')
    model = nn.Sequential()
    name = f'yolo_{order}'
    model.add_module(
        name,
        YOLOLayer(anchors, classes, image_size, name)
    )
    return lastFilter, model
