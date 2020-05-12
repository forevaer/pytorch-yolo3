from torch import nn
from config.config import logRegister
from entity.enum import LayerType
from entity.parser import LayerDefinition
from assist.utils import getFuncName, parseEnumName

# 网络生成器注册中心
creators = {}


# 生成器装饰器
def creator(name: LayerType):
    def register(createFunction):
        if logRegister:
            print(f'register creator           : {parseEnumName(name)} >>> {getFuncName(createFunction)}')
        creators[name] = createFunction

    return register


# 原子生成
def atomCreate(definition, *args, **kwargs):
    name = definition.type
    assert name in creators, f'cannot find creator : {name}'
    return creators[name](definition, *args, **kwargs)


# 外部直接生成器
def create(definitions: list):
    hyperParameters = definitions.pop(0)
    assert isinstance(hyperParameters, LayerDefinition)
    filterCollector = [hyperParameters.intValue('channels')]
    lastFilter = None
    models = nn.ModuleList()
    for idx, definition in enumerate(definitions):
        assert isinstance(definition, LayerDefinition)
        if definition.type is LayerType.YOLO:
            setattr(definition, 'height', hyperParameters.intValue('height'))
        lastFilter, model = atomCreate(definition, idx, filterCollector, lastFilter)
        models.append(model)
        filterCollector.append(lastFilter)
    return hyperParameters, models
