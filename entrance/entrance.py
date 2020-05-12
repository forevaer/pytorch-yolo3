from config import config
from assist.assist import getModule
from registry.loaderRegistry import getLoader
from registry.entranceRegister import getEntrance
from registry.optimizerRegistry import getOptimizerCreator


def main():
    loader = getLoader(config.phase)
    device, model = getModule()
    optimizer = getOptimizerCreator(config.optimizer)(model)
    entrance = getEntrance(config.phase)
    entrance(device, model, loader, optimizer)


if __name__ == '__main__':
    main()
