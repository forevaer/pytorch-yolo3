# YOLO

`YOLOLayer`的定义在[YOLONet.py](../net/YOLONet.py)。<br>
可以看到，`YOLO`的作用，很大程度上都是在计算损失，本身没有可以梯度下降更新的参数。

# 变换
变换网络结构之后，依据[loss](../mds/loss.md)进行计算，计算详情查看[computy.py](../assist/compute.py)
