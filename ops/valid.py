import torch
from config import config
from entity.logger import Logger
from entity.metric import metrics
from entity.viewer import Viewer
from registry.entranceRegister import entrance, PHASE


@entrance(PHASE.VALID)
def Valid(device, model, loader, optimizer):
    model.eval()
    viewer = Viewer() if config.view else None
    with torch.no_grad():
        with Logger(metrics, config.batch_size) as logger:
            for _ in range(config.epochs):
                for idx, (_, images, targets) in enumerate(loader):
                    images = images.to(device)
                    targets = targets.to(device)
                    model(images, targets)
                    yolos = model.yoloLayers
                    logger.log_yolos(yolos)
                    if viewer is not None:
                        viewer.update(yolos)
                    logger.step()
