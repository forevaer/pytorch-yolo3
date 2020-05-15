import numpy as np
from config.config import viewer_limit
from collections import defaultdict
from net.YOLONet import YOLOLayer
from entity.metric import MetricEntity
from matplotlib import pyplot as plt


class Viewer(object):

    def __init__(self, limit=viewer_limit):
        self.limit = limit
        self.collections = {}
        self.count = 0
        self.pop = False
        self.start = 1
        plt.ion()
        plt.figure(figsize=(12, 8))

        """
        {
            yolo_name:{
               loss: {
                 name: []
               }
               acc: {
                 name: []
               }
            }
        }
        """

    def update(self, yolos):

        if not self.pop:
            self.pop = not self.count < self.limit
        for yolo in yolos:
            assert isinstance(yolo, YOLOLayer)
            update_dict = self.collections.setdefault(yolo.name, defaultdict(dict))
            loss_collect = update_dict['loss']
            for entry in yolo.metrics.lossEntries():
                assert isinstance(entry, MetricEntity)
                collect = loss_collect.setdefault(entry.name, [])
                collect.append(entry.value)
                if self.pop:
                    collect.pop(0)
            acc_collect = update_dict['acc']
            for entry in yolo.metrics.accEntries():
                assert isinstance(entry, MetricEntity)
                collect = acc_collect.setdefault(entry.name, [])
                collect.append(entry.value)
                if self.pop:
                    collect.pop(0)
        if not self.pop:
            self.count += 1
        else:
            self.start += 1
        self.draw()

    def draw(self, keep=False):
        plt.clf()
        end = self.start + self.limit - 1 if self.pop else self.count
        index = [x for x in range(self.start, end + 1)]
        for order, yolo_name in enumerate(self.collections):
            loss_ax = plt.subplot(2, 3, order + 1)
            plt.title(f'{yolo_name}-loss')
            yolo_collector = self.collections[yolo_name]
            for loss_key, loss_value in yolo_collector['loss'].items():
                loss_ax.plot(index, loss_value, label=loss_key, marker='^')
            plt.legend(loc=2, prop={'size': 6})

            acc_ax = plt.subplot(2, 3, 4 + order)
            plt.title(f'{yolo_name}-acc')
            for acc_key, acc_value in yolo_collector['acc'].items():
                acc_ax.plot(index, acc_value, label=acc_key, marker='s')
            plt.legend(loc=2, prop={'size': 6})
        plt.pause(0.5)
        if keep:
            plt.show()
        else:
            plt.ioff()
