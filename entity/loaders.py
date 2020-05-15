from config import config
from entity.enum import PHASE
from registry.loaderRegistry import loader
from entity.dataset import ImageSet, ListSet, DataLoader


@loader(PHASE.TRAIN)
def getTrainLoader():
    trainSet = ListSet(config.train_data.replace('coco', 'mini') if config.mini else config.train_data)
    trainLoader = DataLoader(trainSet, shuffle=True, batch_size=config.batch_size, pin_memory=True,
                             collate_fn=trainSet.collate_fn)
    return trainLoader


@loader(PHASE.VALID)
def getValidLoader():
    validSet = ListSet(config.valid_data.replace('coco', 'mini') if config.mini else config.valid_data)
    validLoader = DataLoader(validSet, batch_size=config.batch_size, collate_fn=validSet.collate_fn)
    return validLoader


@loader(PHASE.DETECT)
def getDetectLoader():
    detectSet = ImageSet(config.detect_dir)
    detectLoader = DataLoader(detectSet, batch_size=config.batch_size)
    return detectLoader


@loader(PHASE.WEIGHT2PTH)
def getWeight2PthLoader():
    return None
