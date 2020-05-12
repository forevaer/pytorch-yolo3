import torch
import random
import numpy as np
from PIL import Image
from os.path import basename, join
from torch.nn import functional as F
from assist.utils import ensurePath
from config.config import image_size
from matplotlib.ticker import NullLocator
from matplotlib import patches, pyplot as plt


def getColors():
    cmap = plt.get_cmap("tab20b")
    return [cmap(i) for i in np.linspace(0, 1, 20)]


def createDrawBBoxFunc(ax, point=False):
    def drawBBox(label, x, y, w, h, borderColor='cyan', fontColor='white'):
        if point:
            w -= x
            h -= y
        plt.text(x, y, label, color=fontColor, verticalalignment='top', bbox={'color': borderColor, 'pad': 0})
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=borderColor, facecolor='none')
        ax.add_patch(rect)

    return drawBBox


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def pad2square(src, padValue):
    c, h, w = src.shape
    diff = np.abs(h - w)
    half_pad = diff // 2
    pad1, pad2 = half_pad, diff - half_pad
    pad = (pad1, pad2, 0, 0) if not w < h else (0, 0, pad1, pad2)
    # pad(left, right, top, bottom)
    return F.pad(src, pad, value=padValue), pad


def horizontal_flip(image, target):
    a = target[:, 2]
    a = 1 - a
    target[: 2] = a
    return torch.flip(image, [-1]), target


def readTensorImage(path, transform=None):
    image = Image.open(path).convert('RGB')
    if transform is not None:
        image = transform(image)
    return image


def readNumpyImage(path):
    return np.array(Image.open(path))


class ImageRender(object):

    def __init__(self, src, detectionProcessor=None, labels=None, colors=None, point=False, output=None):
        self.src = src
        self.point = point
        self.detectionProcessor = detectionProcessor
        self.output = output
        plt.figure()
        self.fig, self.ax = plt.subplots(1)
        self.labels = labels
        if output is not None:
            ensurePath(output, False)
        self.savePath = join(output, basename(src))
        self.imageSize = None
        if colors is None:
            self.colors = getColors()
        else:
            self.colors = colors
        self.renderImage()

    def renderImage(self):
        image = readNumpyImage(self.src)
        self.imageSize = image.shape[:2]
        self.ax.imshow(image)

    def renderDetections(self, detections):
        if detections is None:
            print('detection is None')
            return
        if self.detectionProcessor is not None:
            detections = self.detectionProcessor(detections, self.imageSize, image_size)
        unique_labels = detections[:, -1].cpu().unique()
        predict_classes = len(unique_labels)
        bbox_colors = random.sample(self.colors, predict_classes)
        for x, y, w, h, confidence, class_confidence, predict_class in detections:
            int_predict_class = int(predict_class)
            color = bbox_colors[int(np.where(unique_labels == int_predict_class)[0])]
            label = self.labels[int_predict_class]
            self._renderDetection(label, x, y, w, h, borderColor=color)

    def _renderDetection(self, label, x, y, w, h, borderColor, fontColor='white'):
        if self.point:
            w -= x
            h -= y
        plt.text(x, y, label, color=fontColor, verticalalignment='top', bbox={'color': borderColor, 'pad': 0})
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=borderColor, facecolor='none')
        self.ax.add_patch(rect)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(self.savePath, bbox_inches="tight", pad_inches=0.0)
        plt.close()


def standardDraw(image_path, label_path, normalize=True):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    h, w, c = image.shape
    pad = h - w if h > w else w - h
    half = pad // 2
    pad = ((0, 0), (half, pad - half), (0, 0)) if h > w else ((0, 0), (0, 0), (half, pad - half))
    mask = np.pad(image[:], pad, mode='constant', constant_values=0)
    # mask = image
    anchor = np.loadtxt(label_path).reshape(-1, 5)[0][1:]
    if normalize:
        anchor[0] = (anchor[0] - anchor[2] / 2) * w
        anchor[1] = (anchor[1] - anchor[3] / 2) * h
        anchor[2] = anchor[2] * w
        anchor[3] = anchor[3] * h
    anchor[0] += 0 if w > h else half
    anchor[1] += 0 if h > w else half
    print(anchor)
    fig, ax = plt.subplots(1)
    rect = patches.Rectangle((anchor[0], anchor[1]), anchor[2], anchor[3], linewidth=2, edgecolor='cyan',
                             facecolor='red')
    ax.add_patch(rect)
    ax.imshow(mask)
    plt.show()
