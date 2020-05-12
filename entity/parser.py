from os.path import exists
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


class ConfigParser(object):
    def __init__(self, configPath):
        assert exists(configPath), f'file {configPath} does not exists !'
        self.definitions = []
        self.configPath = configPath
        self.lines = None
        self.load()
        self.filter()
        self.parse()

    def load(self):
        """
        加载指定配置文件
        """
        with open(self.configPath, 'r') as f:
            self.lines = f.readlines()

    def filter(self):
        """
        过滤空行以及注释行
        """
        self.lines = [line.strip() for line in self.lines]
        self.lines = [line for line in self.lines if not (line == '' or line.startswith('#'))]

    def parse(self):
        """
        读取每一行内容
        """
        tempDefinition = None
        for line in self.lines:
            if line.startswith('['):
                if tempDefinition is not None:
                    self.definitions.append(tempDefinition)
                tempDefinition = LayerDefinition(line[1:-1])
                continue
            key, value = line.split('=')
            tempDefinition.addValue(key.strip(), value.strip())
        self.definitions.append(tempDefinition)

    def result(self):
        """
        解析结果
        """
        return self.definitions


if __name__ == '__main__':
    definitions = ConfigParser('../config/yolov3.cfg').result()
    for item in definitions:
        print(item)
