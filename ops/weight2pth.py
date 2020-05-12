from assist import assist
from registry.entranceRegister import entrance, PHASE


@entrance(PHASE.WEIGHT2PTH)
def Weight2Pth(device, model, loader, optimizer):
    assist.saveModel(model)
