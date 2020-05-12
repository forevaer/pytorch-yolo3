import torch
from config import config


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def tensor2cpu(tensor, item=True):
    result = tensor.detach().cpu()
    if item:
        result = result.item()
    return result


def wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = w1 * h1 + 1e-16 + w2 * h2 - inter_area
    return inter_area / union_area


def rescale_boxes(boxes, original_shape, standard_size):
    """
     真实的图片会经过三种尺寸变换
         0. 原始尺寸
         1. 正方形填充
         2. 标准尺寸放缩
     结果的bbox是相对于标准尺寸的，要恢复到原图，需要进行如下步骤
         1. 放缩到正方形尺寸
         2. 去除边框填充偏移
         3. 真实bbox
     不过，通过归一化的方式，可以这样操作
         0. 剔除边框填充
         1. 等比例缩放原图
         2. 真实bbox
    """
    origin_h, origin_w = original_shape
    # 填充为正方形的大小
    paddedLength = origin_h if origin_h > origin_w else origin_w
    # 计算缩放时的填充
    pad_x = (paddedLength - origin_w) * (standard_size / paddedLength)
    pad_y = (paddedLength - origin_h) * (standard_size / paddedLength)
    # 获得无填充缩放大图
    un_pad_h = standard_size - pad_y
    un_pad_w = standard_size - pad_x
    # 大图上的bbox直接缩放
    boxes[:, 0] = (boxes[:, 0] - pad_x // 2) / un_pad_w * origin_w
    boxes[:, 1] = (boxes[:, 1] - pad_y // 2) / un_pad_h * origin_h
    boxes[:, 2] = (boxes[:, 2] - pad_x // 2) / un_pad_w * origin_w
    boxes[:, 3] = (boxes[:, 3] - pad_y // 2) / un_pad_h * origin_h
    # 尺寸超标问题
    boxes[boxes < 0] = 0
    max_x = boxes[:, 2]
    max_y = boxes[:, 3]
    max_x[max_x > origin_w] = origin_w
    max_y[max_y > origin_h] = origin_h
    boxes[:, 2] = max_x
    boxes[:, 3] = max_y
    return boxes


def bbox_iou(box1, box2, point=True):
    if not point:
        # (x, y, w, h)
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # (x1, y1, x2, y2)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # clamp:区域限制
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # iou包围问题
    mask = (inter_area == b1_area) | (inter_area == b2_area)
    ious = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    ious[mask] = 1
    return ious


def NMS(prediction):
    #  (center_x, center_y, width, height) ==> (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_idx, predict_image in enumerate(prediction):
        # 筛选置信度大于confidence_threshold的数据
        predict_image = predict_image[predict_image[:, 4] >= config.confidence_threshold]
        # 不存在就下一组
        if not predict_image.size(0):
            continue
        # 置信度 x 类别， 作为总score
        score = predict_image[:, 4] * predict_image[:, 5:].max(1)[0]
        # 置信度从大到小排列
        predict_image = predict_image[(-score).argsort()]
        # 最大的score，和类别
        class_confidence, predict_class = predict_image[:, 5:].max(1, keepdim=True)
        # 把结果进行拼接
        detections = torch.cat((predict_image[:, :5], class_confidence.float(), predict_class.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        # print("=========================")
        while detections.size(0):
            # 广播操作
            # 计算第一个和其他全部的iou
            ious = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4], point=True)
            # print(ious)
            large_overlap = ious > config.nms_threshold
            # print('overlap', large_overlap)
            # 找到标签统一匹配的
            label_match = detections[0, -1] == detections[:, -1]
            # print('label', label_match)
            # 必须同时满足两者，避免过滤临近框不同类别
            invalid = large_overlap & label_match
            # print('invalid', invalid)
            # 置信度
            weights = detections[invalid, 4:5]
            # 把多个相同的类型框进行坐标平均
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            # 把检测的最大的加上
            keep_boxes += [detections[0]]
            # 把未匹配的继续运算
            detections = detections[~invalid]
        # 如果检测出一些
        if keep_boxes:
            # 添加
            output[image_idx] = torch.stack(keep_boxes)
        # print(output)
    return output


def build_targets(predict_boxes, predict_class, predict_anchors, target, ignore_threshold):
    # type support
    ByteTensor = torch.cuda.ByteTensor if predict_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if predict_boxes.is_cuda else torch.FloatTensor
    # 参数纬度获取
    batch_count = predict_boxes.size(0)
    anchor_count = predict_boxes.size(1)
    grid = predict_boxes.size(2)
    # 预测类别
    classes = predict_class.size(-1)
    # 结果容器构造

    # 全图mask标记
    # 真实背景
    reality_background = ByteTensor(batch_count, anchor_count, grid, grid).fill_(1)
    # 真实全景
    reality_foreground = ByteTensor(batch_count, anchor_count, grid, grid).fill_(0)
    # 真实分类
    class_acc = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    # 真实iou
    iou_scores = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)

    # 真实差距
    tx = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    ty = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    tw = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    th = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    # 真实分类差距
    tc = FloatTensor(batch_count, anchor_count, grid, grid, classes).fill_(0)

    # 归一化位置反归一化
    # 把真实的框映射到当前的featureMap上，后续进行对比
    target_boxes = target[:, 2:6] * grid
    # 真实位置
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # iou，当前featureMap上的iou
    ious = torch.stack([wh_iou(anchor, gwh) for anchor in predict_anchors])
    # 分值，索引
    best_iou_score, best_iou_idx = ious.max(0)

    # 原来细节数据都自成纬度，t()之后转到同一纬度，然后分离
    target_batch, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    # 1. 原始坐标为浮点数，网格中只能是整数
    # 2. 向量纬度顺序为(y, x), (j, i) <=> (int(y), int(x)) 等同于真实坐标(x, y)
    gi, gj = gxy.long().t()

    # 更新全图掩层
    # 真实前景
    reality_foreground[target_batch, best_iou_idx, gj, gi] = 1
    # 真实背景
    reality_background[target_batch, best_iou_idx, gj, gi] = 0
    # 置信度高的也标记对象
    for i, anchor_iou in enumerate(ious.t()):
        reality_background[target_batch[i], anchor_iou > ignore_threshold, gj[i], gi[i]] = 0

    # 对应坐标更新差值
    # 真实框图的坐标残差
    tx[target_batch, best_iou_idx, gj, gi] = gx - gx.floor()
    ty[target_batch, best_iou_idx, gj, gi] = gy - gy.floor()
    # 宽高使用对数比例进行更新
    # 真实框图的宽高对数分值
    tw[target_batch, best_iou_idx, gj, gi] = torch.log(gw / predict_anchors[best_iou_idx][:, 0] + 1e-32)
    th[target_batch, best_iou_idx, gj, gi] = torch.log(gh / predict_anchors[best_iou_idx][:, 1] + 1e-32)
    # 指定类别更新
    tc[target_batch, best_iou_idx, gj, gi, target_labels] = 1

    # 预测结果对比更新类型
    class_acc[target_batch, best_iou_idx, gj, gi] = (
            predict_class[target_batch, best_iou_idx, gj, gi].argmax(-1) == target_labels).float()
    # 对应的iou计算
    iou_scores[target_batch, best_iou_idx, gj, gi] = bbox_iou(predict_boxes[target_batch, best_iou_idx, gj, gi],
                                                              target_boxes, point=False)
    # 对象掩层置信度
    reality_object = reality_foreground.float()
    return iou_scores, class_acc, reality_foreground.bool(), reality_background.bool(), tx, ty, tw, th, tc, reality_object
