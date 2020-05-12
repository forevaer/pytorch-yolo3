import torch
from config import config
from entity.logger import Logger
from entity.metric import metrics
from registry.entranceRegister import entrance, PHASE


@entrance(PHASE.VALID)
def Valid(device, model, loader, optimizer):
    model.eval()
    with torch.no_grad():
        with Logger(metrics, config.batch_size) as logger:
            for idx, (_, images, targets) in enumerate(loader):
                images = images.to(device)
                targets = targets.to(device)
                model(images, targets)
                for yolo in model.yoloLayers:
                    logger.log(yolo.metrics.line())
                logger.step()
