import torch
from config import config
from assist import assist
from entity.logger import Logger
from entity.metric import metrics
from entity.viewer import Viewer
from registry.entranceRegister import entrance, PHASE


@entrance(PHASE.TRAIN)
def Train(device, model, loader, optimizer):
    viewer = Viewer() if config.view else None
    with Logger(metrics, config.batch_size) as logger:
        for epoch in range(config.epochs):
            model.train()
            for batch_idx, (_, images, targets) in enumerate(loader):
                assert isinstance(images, torch.Tensor)
                images = images.to(device)
                targets = targets.to(device)
                loss, output = model(images, targets)
                loss.backward()
                if batch_idx % config.model_save_interval == 0:
                    assist.saveModel(model)
                yolos = model.yoloLayers
                logger.log_yolos(yolos)
                if viewer is not None:
                    viewer.update(yolos)
                logger.step()
                optimizer.step()
                optimizer.zero_grad()
