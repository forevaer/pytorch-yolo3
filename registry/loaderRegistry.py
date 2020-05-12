from entity.enum import PHASE
from config.config import logRegister
from assist.utils import getFuncName, parseEnumName

loaders = {}


def loader(name):
    def loaderFunc(function):
        if logRegister:
            print(f'register loader            : {parseEnumName(name)} >>> {getFuncName(function)}')
        loaders[name] = function

    return loaderFunc


def getLoader(phase: PHASE):
    return loaders[phase]()
