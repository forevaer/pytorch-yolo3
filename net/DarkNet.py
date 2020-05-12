import torch
import numpy as np
from torch import nn
from assist.compute import tensor2cpu
from registry.creatorRegistry import create
from entity.parser import ConfigParser, LayerDefinition


class DarkNet(nn.Module):

    def __init__(self, config_path, image_size=416):
        super(DarkNet, self).__init__()
        self.module_definitions = ConfigParser(config_path).result()
        self.hyperParams, self.moduleList = create(self.module_definitions)
        self.yoloLayers = [layer[0] for layer in self.moduleList if hasattr(layer[0], 'metrics')]
        self.imageSize = image_size
        self.seen = 0
        self.headInfo = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, target=None):
        image_size = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (definition, module) in enumerate(zip(self.module_definitions, self.moduleList)):
            assert isinstance(definition, LayerDefinition), f"illegal type: need LayerDefinition but {type(definition)}"
            if definition.type.isBasic():
                x = module(x)
            elif definition.type.isRoute():
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in definition.stringValue('layers').split(',')],
                              1)
            elif definition.type.isShortCut():
                layer_i = definition.intValue('from')
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif definition.type.isYOLO():
                x, layer_loss = module[0](x, target, image_size)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = tensor2cpu(torch.cat(yolo_outputs, 1), item=False)
        return yolo_outputs if target is None else (loss, yolo_outputs)

    def save_weights(self, path, cutOff=-1):
        self.headInfo[3] = self.seen
        with open(path, 'wb') as file:
            self.headInfo.tofile(file)
            for i, (definition, module) in enumerate(zip(self.module_definitions, self.moduleList)):
                assert isinstance(definition, LayerDefinition)
                if definition.type.isConv():
                    conv_layer = module[0]
                    if definition.boolValue('batch_normalize'):
                        bn_layer = module[1]
                        assert isinstance(bn_layer, nn.BatchNorm2d)
                        bn_layer.bias.data.cpu().numpy().tofile(file)
                        bn_layer.weight.data.cpu().numpy().tofile(file)
                        bn_layer.running_mean.data.cup().numpy().tofile(file)
                        bn_layer.running_var.data.cpu().numpy().tofile(file)
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(file)
                    conv_layer.weight.data.cpu().numpy().tofile(file)

    def load_weight(self, path):
        with open(path, 'rb') as f:
            self.headInfo = np.fromfile(f, dtype=np.int32, count=5)
            self.seen = self.headInfo[3]
            weights = np.fromfile(f, dtype=np.float32)
        curOff = None
        if "darknet53.conv.74" in path:
            curOff = 75

        ptr = 0
        for i, (definition, module) in enumerate(zip(self.module_definitions, self.moduleList)):
            if i == curOff:
                break
            assert isinstance(definition, LayerDefinition)
            if definition.type.isConv():
                conv_layer = module[0]
                if definition.boolValue('batch_normalize'):
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                    # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
