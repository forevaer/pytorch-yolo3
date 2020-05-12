from torch import optim, nn
from registry.optimizerRegistry import optimizerCreator, OPTIMIZER


@optimizerCreator(OPTIMIZER.ADAM)
def adamOptimizer(model: nn.Module):
    return optim.Adam(model.parameters())


@optimizerCreator(OPTIMIZER.SGD)
def sgdOptimizer(model: nn.Module):
    return optim.SGD(model.parameters())
