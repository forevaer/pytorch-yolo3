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

    def isNet(self):
        return self == LayerType.NET

    def isConv(self):
        return self == LayerType.CONVOLUTIONAL

    def isShortCut(self):
        return self == LayerType.SHORTCUT

    def isRoute(self):
        return self == LayerType.ROUTE

    def isUpSample(self):
        return self == LayerType.UPSAMPLE

    def isYOLO(self):
        return self == LayerType.YOLO

    def isMaxPooling(self):
        return self == LayerType.MAXPOOLING

    def isBasic(self):
        return self in [
            LayerType.MAXPOOLING,
            LayerType.CONVOLUTIONAL,
            LayerType.UPSAMPLE
        ]


@unique
class OPTIMIZER(Enum):
    ADAM = 'adam'
    SGD = 'sgd'


@unique
class PHASE(Enum):
    TRAIN = 'train'
    VALID = "valid"
    DETECT = 'detect'
    WEIGHT2PTH = 'weight2pth'
