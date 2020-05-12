import torch
from torch import nn
from entity.metric import Metric
from assist.compute import build_targets, tensor2cpu
from config.config import weights_background, weights_foreground, weights_h, weights_w, weights_x, weights_y, weights_class


class YOLOLayer(nn.Module):
    def __init__(self, anchors, classes, image_size=416, name=None):
        super(YOLOLayer, self).__init__()
        self.name = name
        self.metrics = Metric()
        self.anchors = anchors
        self.anchor_count = len(anchors)
        self.classes = classes
        self.ignore_threshold = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.grid_size = 0
        self.image_size = image_size
        self.stride = None
        self.grid_x = None
        self.grid_y = None
        self.anchor_w = None
        self.anchor_h = None
        self.scale_anchors = None
        self.initName()
        #
        self.x_weight = weights_x
        self.y_weight = weights_y
        self.w_weight = weights_w
        self.h_weight = weights_h
        self.c_weight = weights_class
        self.foreground_weight = weights_foreground
        self.background_weight = weights_background

    def initName(self):
        self.metrics.setMetric('name', self.name)

    def compute_grid(self, grid_size, cuda=False):
        self.grid_size = grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # 方格划分，可以视为新像素
        self.stride = self.image_size / self.grid_size

        # 高度
        g = self.grid_size
        # repeat(g x g)图片尺寸
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        # t()，转置即为y
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        # anchor只有长宽，具体的目标点回归学习， 原来步长调整为网格步长
        self.scale_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        # 1. 外部升纬
        # 2. 数量
        # 3. 每个w、h都作为回归目标，各自占领一个纬度
        self.anchor_w = self.scale_anchors[:, 0:1].view((1, self.anchor_count, 1, 1))
        self.anchor_h = self.scale_anchors[:, 1:2].view((1, self.anchor_count, 1, 1))

    def forward(self, x, target, image_size):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        self.image_size = image_size
        # x(count, channel, w x h)
        sample_count = x.size(0)
        grid_size = x.size(2)
        # 把具体细分的类别放到最小层
        prediction = x.view(sample_count, self.anchor_count, self.classes + 5, grid_size, grid_size) \
            .permute(0, 1, 3, 4, 2) \
            .contiguous()

        # 坐标偏移
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 宽高比例
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        # 前景预测
        predict_object = torch.sigmoid(prediction[..., 4])
        # 类别预测
        predict_class = torch.sigmoid(prediction[..., 5:])

        if grid_size != self.grid_size:
            self.compute_grid(grid_size, cuda=x.is_cuda)

        predict_boxes = FloatTensor(prediction[..., :4].shape)
        # 加上偏移
        predict_boxes[..., 0] = x.data + self.grid_x
        predict_boxes[..., 1] = y.data + self.grid_y
        # 乘以比例
        predict_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        predict_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat((
            # bbox
            predict_boxes.view(sample_count, -1, 4) * self.stride,
            # confidence
            predict_object.view(sample_count, -1, 1),
            # classify
            predict_class.view(sample_count, -1, self.classes)
        ),
            -1
        )

        if target is None:
            return output, 0
        iou_scores, _class_acc, reality_foreground, reality_background, tx, ty, tw, th, tc, reality_object = build_targets(
            predict_boxes=predict_boxes,
            predict_class=predict_class,
            target=target,
            predict_anchors=self.scale_anchors,
            ignore_threshold=self.ignore_threshold
        )
        # 框图损失
        # 预测的是anchor在该featureMap上的变换大小残差
        loss_x = self.x_weight * self.mse_loss(x[reality_foreground], tx[reality_foreground])
        loss_y = self.y_weight * self.mse_loss(y[reality_foreground], ty[reality_foreground])
        loss_w = self.w_weight * self.mse_loss(w[reality_foreground], tw[reality_foreground])
        loss_h = self.h_weight * self.mse_loss(h[reality_foreground], th[reality_foreground])
        # 前背景损失
        # 前景：目标检测
        # 背景：背景检测
        # 都是目标检测的正反例
        loss_foreground = self.bce_loss(predict_object[reality_foreground], reality_object[reality_foreground])
        loss_background = self.bce_loss(predict_object[reality_background], reality_object[reality_background])
        loss_confidence = self.foreground_weight * loss_foreground + self.background_weight * loss_background
        # 分类损失
        loss_class = self.c_weight * self.bce_loss(predict_class[reality_foreground], tc[reality_foreground])
        # 总损失
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_class + loss_confidence

        # 指标
        # 类型判断召回
        class_acc = 100 * _class_acc[reality_foreground].mean()
        # 前景召回
        foreground_acc = predict_object[reality_foreground].mean()
        # 背景召回
        background_acc = predict_object[reality_background].mean()

        self.save_metrics('loss', total_loss)
        self.save_metrics('loss_x', loss_x)
        self.save_metrics('loss_y', loss_y)
        self.save_metrics('loss_w', loss_w)
        self.save_metrics('loss_h', loss_h)
        self.save_metrics('loss_confidence', loss_confidence)
        self.save_metrics('loss_class', loss_class)
        self.save_metrics('class_acc', class_acc)
        self.save_metrics('foreground_acc', foreground_acc)
        self.save_metrics('background_acc', background_acc)
        self.save_metrics('grid', grid_size, tensor=False)

        return output, total_loss

    def save_metrics(self, name, value, tensor=True):
        if tensor:
            value = tensor2cpu(value)
        self.metrics.setMetric(name, value)
