import torch
from assist.compute import NMS
from assist.utils import getLabels
from torch.autograd import Variable
from config.config import output_path
from assist.image import ImageRender
from assist.compute import rescale_boxes
from registry.entranceRegister import entrance, PHASE


@entrance(PHASE.DETECT)
def Detect(device, model, loader, optimizer):
    model.eval()
    detectImagePaths = []
    detectionResults = []
    with torch.no_grad():
        for batch_idx, (image_paths, images) in enumerate(loader):
            images = images.to(device)
            detections = model(images)
            detections = NMS(detections)
            detectImagePaths.extend(image_paths)
            detectionResults.extend(detections)
    labels = getLabels()
    for image_idx, (image_path, detection) in enumerate(zip(detectImagePaths, detectionResults)):
        with ImageRender(image_path, point=True, labels=labels, detectionProcessor=rescale_boxes, output=output_path) as render:
            render.renderDetections(detection)

