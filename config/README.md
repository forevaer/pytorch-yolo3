# 配置说明

# 运行模式
基础的模式定义有如下四种
- `TRAIN`
- `VALID`
- `DETECT`
- `WEIGHT2PTH`

为了保留自我操作的空间，启用`fewTrain`会更换自定义的小数据集进行训练，目录位于[few](../data/mini)。<br>

训练的随时恢复，通过`initFromWeight`来进行控制，当模式为`WEIGHT2PTH`，单纯进行权重转`pth`文件。

后续配置其他地方会进行介绍。
