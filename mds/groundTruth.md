# 边框
这里主要说明`groundTruth`的变换过程。综述参看[损失说明](loss.md)。<br>
详细代码参看[utils.py](../assist/compute.py)中的`create_target`方法。<br>
这里只做拆解解析。

# 基础准备
方法的很大一部分都是在进行容器准备，后续进行参数更新。<br>
因此，前半截需要注意的只是
- 纬度
- 初始值

```python
 ByteTensor = torch.cuda.ByteTensor if predict_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if predict_boxes.is_cuda else torch.FloatTensor
    # 参数纬度获取
    batch_count, anchor_count, grid = predict_boxes.size(0), predict_boxes.size(1), predict_boxes.size(2)
    classes = predict_class.size(-1)
    # 结果容器构造

    # 全图mask标记
    # 默认全背景
    reality_background = ByteTensor(batch_count, anchor_count, grid, grid).fill_(1)
    # 默认无对象
    reality_foreground = ByteTensor(batch_count, anchor_count, grid, grid).fill_(0)
    # 默认无分类
    class_index = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    # 默认无分值
    iou_scores = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)

    # 预测框
    tx = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    ty = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    tw = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    th = FloatTensor(batch_count, anchor_count, grid, grid).fill_(0)
    # 预测分类
    tc = FloatTensor(batch_count, anchor_count, grid, grid, classes).fill_(0)
```

# anchor
```python
target_boxes = target[:, 2:6] * grid
```

这里可以很明白的去理解：**`anchor`的变换是不需要回归的**。
归一之后的`bbox`，能够在任意尺寸的`FeatureMap`上进行复现。

# 坐标
```python
 gi, gj = gxy.long().t()
```
- 纬度

因为大部分数据都需要进行回归，所以都自我占据一个纬度，即使是`(x,y)`，也会这样`[[x],[y]]`。<br>
因此，代码中常见`.t()`操作，这是为了把纬度合一，方便取值 `[[x],[y]].t() ==> [[x,y]]`。

- 索引

正常的像素坐标都是`(x,y)`，但是`Tensor`的元素取值，可以看作是`(y,x)`, 因此可以看到后续使用的坐标都是`(gj, gi)` .


# 掩层

主要用作`前背景`分类依据，对于`FeatureMap`上的每个点进行归类。<br>
前景的掩版，默认都是0， 背景的掩版，默认都是1。<br>
真实的情况应该两个向量相加得到单位`ones`。<br>
不过计算当中，前景掩版一定是预测的，但是背景掩版在指定的`iou`阈值上也会进行前景标记。

# 框图
```python
    tx[target_batch, best_iou_idx, gj, gi] = gx - gx.floor()
    ty[target_batch, best_iou_idx, gj, gi] = gy - gy.floor()
    # 宽高使用对数比例进行更新
    # 真实框图的宽高对数分值
    tw[target_batch, best_iou_idx, gj, gi] = torch.log(gw / predict_anchors[best_iou_idx][:, 0] + 1e-16)
    th[target_batch, best_iou_idx, gj, gi] = torch.log(gh / predict_anchors[best_iou_idx][:, 1] + 1e-16)
```
其中坐标点为小数部分差值，宽高为比例对数。<br>
得到的是真实的映射`bbox`。

# 其他
```python
    class_acc[target_batch, best_iou_idx, gj, gi] = (predict_class[target_batch, best_iou_idx, gj, gi].argmax(-1) == target_labels).float()
    # 对应的iou计算
    iou_scores[target_batch, best_iou_idx, gj, gi] = bbox_iou(predict_boxes[target_batch, best_iou_idx, gj, gi], target_boxes, point=False)
    # 对象掩层置信度
    reality_object = reality_foreground.float()
```

# 后续
这里传入的数据，主要是因为预测决定了`FeatureMap`的尺寸，需要计算真实的框图在`FeatureMap`上的映射，方便后续对比损失。<br>
其次，这里的计算都依赖于预测边框`bbox`，但是不停的迭代，会越发精准，后续还需要进行过滤操作。