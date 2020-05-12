# 工程说明

- [assist](../assist): 辅助工具
  - [assist.py](../assist/assist.py) : 入口启动协助办法
  - [compute.py](../assist/compute.py): 网络核心运算
  - [image.py](../assist/image.py): 图像操作
  - [utils.py](../assist/utils.py): 其他操作 
- [config](../config/README.md) : 配置说明
- [data](../data/README.md) : 数据集合
- [entity](../entity/README.md): 实体定义
- [entrance](../entrance/entrance.py): 启动入口，修改配置即可
- [registry](../registry/README.md): 注册中心
- ops
  - [train.py](../ops/train.py): 训练
  - [valid.py](../ops/valid.py): 验证
  - [detect.py](../ops/detect.py): 检测
  - [weight2pth.py](../ops/weight2pth.py):  `weight`文件转`pt`
- pts: 模型文件`model.pth`存放目录
- output: `detect` 文件输出目录
