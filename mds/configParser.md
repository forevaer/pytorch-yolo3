# config
`YOLO`网络结构53层，但是在网络定义中都看不到如此繁杂的网络，使用的都是原作者的基础配置。<br>
详细内容可以查看[yolov3.cfg](../config/yolov3.cfg)。<br>
原来的代码中，太多的字符串，感觉太杂乱，因此学习并整理一下。

# 类型定义
网络的配置文件当中，存在多种类型的定义。原来代码直接通过字符串比较比较难看，在此定义为了枚举类型。<br>
```python
from enum import Enum, unique

@unique
class LayerType(Enum):
    """
    配置类别
    """
    NET = 'net'
    CONVOLUTIONAL = 'convolutional'
    SHORTCUT = 'shortcut'
    ROUTE = 'route'
    UPSAMPLE = 'upsample'
    YOLO = 'yolo'
    MAXPOOLING = 'maxpooling'
```
详细定义可以查看[enum.py](../entity/enum.py)

# 定义信息
原始的信息解析，都是直接通过字典进行存取，不好看而且不统一，在此专门定义了一个类专门进行管理。
```python
from entity.enum import LayerType

class LayerDefinition(object):
    """
    配置属性
    """
    def __init__(self, cls):
        self.type = LayerType(cls)

    def addValue(self, name, value):
        """
        注册属性
        """
        setattr(self, name, value)

    def getvalue(self, name, cls=None):
        """
        获取属性
        """
        if not hasattr(self, name):
            return None
        value = getattr(self, name)
        if cls is not None:
            return cls(value)
        return value

    def stringValue(self, name) -> str:
        return self.getvalue(name, str)

    def intValue(self, name):
        return self.getvalue(name, int)

    def boolValue(self, name):
        return self.getvalue(name, bool)

    def __str__(self):
        return str(self.__dict__)
```
内部也可以通过字典方式进行实现，为了简洁，采用的属性设置。

# 文件解析
`python`的文件解析十分简便，全部读取之后，具体解析应该注意这几方面
- 忽略注释行
- 去除空白字符
- 注意单元配置

详细实现请查看[configParser](../entity/parser.py)，其中专门分列方法对各操作进行了实现。<br>
直接运行解析信息如下
```text
{'type': <LayerType.NET: 'net'>, 'batch': '16', 'subdivisions': '1', 'width': '416', 'height': '416', 'channels': '3', 'momentum': '0.9', 'decay': '0.0005', 'angle': '0', 'saturation': '1.5', 'exposure': '1.5', 'hue': '.1', 'learning_rate': '0.001', 'burn_in': '1000', 'max_batches': '500200', 'policy': 'steps', 'steps': '400000,450000', 'scales': '.1,.1'}
{'type': <LayerType.CONVOLUTIONAL: 'convolutional'>, 'batch_normalize': '1', 'filters': '32', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}
{'type': <LayerType.CONVOLUTIONAL: 'convolutional'>, 'batch_normalize': '1', 'filters': '64', 'size': '3', 'stride': '2', 'pad': '1', 'activation': 'leaky'}
{'type': <LayerType.CONVOLUTIONAL: 'convolutional'>, 'batch_normalize': '1', 'filters': '32', 'size': '1', 'stride': '1', 'pad': '1', 'activation': 'leaky'}
{'type': <LayerType.CONVOLUTIONAL: 'convolutional'>, 'batch_normalize': '1', 'filters': '64', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'}
{'type': <LayerType.SHORTCUT: 'shortcut'>, 'from': '-3', 'activation': 'linear'}
{'type': <LayerType.CONVOLUTIONAL: 'convolutional'>, 'batch_normalize': '1', 'filters': '128', 'size': '3', 'stride': '2', 'pad': '1', 'activation': 'leaky'}
{'type': <LayerType.CONVOLUTIONAL: 'convolutional'>, 'batch_normalize': '1', 'filters': '64', 'size': '1', 'stride': '1', 'pad': '1', 'activation': 'leaky'}
{'type': <LayerType.CONVOLUTIONAL: 'convolutional'>, 'batch_normalize': '1', 'filters': '128', 'size': '3', 'stride': '1', 'pad': '1', 'activ
...
```
类型信息具体保留，后续判断很多。
