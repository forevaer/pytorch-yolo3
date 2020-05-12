from entity.enum import PHASE
from config.config import logRegister
from assist.utils import getFuncName, parseEnumName

entrances = {}


def entrance(name: PHASE):
    def entranceRegister(function):
        if logRegister:
            print(f'register entrance          : {parseEnumName(name)} >>> {getFuncName(function)}')
        entrances[name] = function

    return entranceRegister


def getEntrance(name: PHASE):
    return entrances[name]
