# 网络生成
原先的网络生成模块都交杂在一块，上下文贯穿但是各种逻辑分离不清晰。<br>
这里专门设计了针对的模块生成器进行生成，只需要将生成器进行注册，后续就能够适应性的采取不同生成器进行模块的生成。

# 生成器中心
```python
from entity.enum import LayerType

# 网络生成器注册中心
creators = {}


# 生成器装饰器
def creator(name: LayerType):
    def register(createFunction):
        creators[name] = createFunction

    return register


# 原子生成
def atomCreate(name: LayerType, *args, **kwargs):
    assert name in creators, f'cannot find creator : {name}'
    return creators[name](*args, **kwargs)
```
- 使用`@creator`进行生成器的注册
- 后续直接使用`atomCreate`进行子网络模块的生成。

# 对外模块生成
```python
# 外部直接生成器
def create(definitions: list):
    hyperParameters = definitions.pop(0)
    assert isinstance(hyperParameters, LayerDefinition)
    filters = [hyperParameters.intValue('channels')]
    models = nn.ModuleList()
    for idx, definition in enumerate(definitions):
        assert isinstance(definition, LayerDefinition)
        if definition.type is LayerType.YOLO:
            setattr(definition, 'height', hyperParameters.intValue('height'))
        _filter, model = atomCreate(definition, idx, filters)
        models.append(model)
        if _filter is not None:
            filters.append(_filter)
    return hyperParameters, models
```
外部直接通过传入解析好的模块配置信息，即可进行网络模型生成。

# 参数说明
- `definition`: 子模块定义类对象
- `order`: 模块顺序，主要用于命名
- `filters`: 输出通道记录，用于调整后续模块参数

# 辅助网络
```python
from torch import nn


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()
```
定义空网络，用于辅助操作。

# 各模块说明

## 卷积层
```python
@creator(LayerType.CONVOLUTIONAL)
def ConvolutionalCreator( definition: LayerDefinition, order: int, filters: list):
    bn = definition.intValue('batch_normalize')
    out_channel = definition.intValue('filters')
    kernel_size = definition.intValue('size')
    stride = definition.intValue('stride')
    padding = (kernel_size - 1) // 2
    model = nn.Sequential()
    model.add_module(
        f'conv_{order}',
        nn.Conv2d(
            in_channels=filters[-1],
            out_channels=out_channel,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=not bn
        )
    )
    if bn:
        model.add_module(
            f'bn_{order}',
            nn.BatchNorm2d(out_channel, momentum=0.9, eps=1e-5)
        )
    if 'leaky' == definition.stringValue('activation'):
        model.add_module(
            f'leaky_{order}',
            nn.LeakyReLU(0.1)
        )
    return None, model
```
基础的卷积层参数都由外部决定，其中有几点策略如下
- 填充策略 ：默认的填充为`padding=(kernel_size - 1) // 2`
- 偏置策略 ：当使用`bn`层的时候，不设偏置
- 激活策略 ：激活函数只使用了`leakyReLU`

卷积、BN、ReLU，一般的卷积层操作都是如此。

## 池化层
```python
@creator(LayerType.MAXPOOLING)
def MaxPoolingCreator(definition: LayerDefinition, order: int, filters: list):
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
    return None, model
```
标准的池化操作，只有当`kernel_size==2 and stride == 1`时候，进行了0填充，具体参数为`(0, 1, 0, 1)`.

## 上采样
```python
# 上采样的生成器
@creator(LayerType.UPSAMPLE)
def UpSampleCreator(definition: LayerDefinition, order: int, filters: list):
    stride = definition.intValue('stride')
    model = nn.Sequential()
    model.add_module(
        f'upsample_{order}',
        nn.Upsample(scale_factor=stride, mode='nearest')
    )
    return None, model
```
原文中是使用方法自己定义，这里直接采用封装好的网络，直接设计即可。

## 路由层
```python
# 路由层的生成器
@creator(LayerType.ROUTE)
def RouteCreator(definition: LayerDefinition, order: int, filters: list):
    layers = [int(x.strip()) for x in definition.stringValue('layers').split(',')]
    filters = sum([filters[i+1] for i in layers])
    model = nn.Sequential()
    model.add_module(
        f'route_{order}',
        EmptyLayer()
    )
    return filters, model
```
内容为空网络，主体作用为计算前面指定层的输出通道之和。

## 快照层
```python
# 快照层生成器
@creator(LayerType.SHORTCUT)
def ShortcutCreator(definition: LayerDefinition, order: int, filters: list):
    filters = filters[definition.intValue('from') + 1]
    model = nn.Sequential()
    model.add_module(
        f'shortcut_{order}',
        EmptyLayer()
    )
    return filters, model
```
获取指定层输出通道数

## YOLO
```python
# YOLO层生成器
@creator(LayerType.YOLO)
def YOLOCreator(definition: LayerDefinition, order: int, filters: list):
    anchor_idx = [int(x.strip()) for x in definition.stringValue('mask').split(',')]
    anchors = [int(x.strip()) for x in definition.stringValue('anchors').split(',')]
    anchors = np.array(anchors).reshape(-1, 2)[anchor_idx].tolist()
    classes = definition.intValue('classes')
    image_size = definition.intValue('height')
    model = nn.Sequential()
    model.add_module(
        f'yolo_{order}',
        YOLOLayer(anchors, classes, image_size)
    )
    return None, model
```
实例化`YOLO`网络，具体结构定义后续跟进，具体文件可以先查看[YOLONet.py](../net/YOLONet.py)。
