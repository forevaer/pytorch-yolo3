from entity.enum import OPTIMIZER
from config.config import logRegister
from assist.utils import getFuncName, parseEnumName

optimizers = {}


def optimizerCreator(name):

    def register(function):
        if logRegister:
            print(f'register optimizerCreator  : {parseEnumName(name)} >>> {getFuncName(function)}')
        optimizers[name] = function

    return register


def getOptimizerCreator(name: OPTIMIZER):
    return optimizers[name]
