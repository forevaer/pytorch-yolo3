import torch
from config import config
from entity.logger import Logger
from net.DarkNet import DarkNet
from assist.utils import weights_init_normal, ensurePath
from registry import optimizerRegistry


def getLogger():
    return Logger()


def getDevice():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    return device


def getModule():
    device = getDevice()
    model = DarkNet(config.model_define_path).to(device)
    model.apply(weights_init_normal)
    if config.initFromWeight:
        model.load_weight(config.model_weights)
    else:
        model.load_state_dict(torch.load(config.model_save_path))
    if config.logModel:
        print(model)
    return device, model


def getOptimizer(model):
    optimizerCreator = optimizerRegistry.getOptimizerCreator(config.optimizer)
    return optimizerCreator(model)


def saveModel(model: torch.nn.Module):
    ensurePath(config.model_save_path)
    torch.save(model.state_dict(), config.model_save_path)


