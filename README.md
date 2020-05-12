# yolo_v3

- [配置说明](config/README.md)
- [工程结构](mds/construct.md)
- [网络解析](./mds/configParser.md)
- [注册中心](./registry/README.md)

# 其他说明

- [损失](mds/loss.md)
- [暗网](mds/DarkNet.md)
- [变换](mds/groundTruth.md)

# 图像

图像具体操作可以参看[compute.py](assist/compute.py)。<br>
其中包含了图像尺寸变换、`NMS`，`IOU`等核心计算方式。

图像统一处理变换为
- 正方形填充
-  尺寸缩放

anchor变换为
- 填充裁剪
- 图形归一化恢复

# anchor

标注格式为`(center_x, center_y, w, h)`，且为归一化的值。<br>
预测坐标为`(x_1, y_1, x_2, y_2)`，且在标准的`(416 x 416)`的尺度上，需要进行坐标转换。<br>

> 实际标注格式为`(class, center_x, center_y, w, h)`，取`bbox`时候需要注意。<br>
> 标注的解析可以查看[dataset.py](entity/dataset.py)，里面有解析办法.<br>
> 预测`bbox`的转换，逻辑计算和坐标转换都在[compute.py](assist/compute.py)
