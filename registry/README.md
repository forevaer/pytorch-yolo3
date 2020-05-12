# 注册中心

由于工程中存在大量重复的同类型操作，因此在这里设计注册中心，关键流程使用相同的一套流程即可，底层自动适配。

# optimizer

适配优化器选择，自动根据选择获取优化器。
> 具体注册请查看[loader.py](../entity/loaders.py)

# loader

获取数据集统一接口，自适应获取数据。

> 注册详情请查看[creators.py](../entity/loaders.py)


# entrance

入口方法分支选择麻烦，同时也进行了注册。

>  [pts](../pts)

# creators

网络配置解析器，详情见[配置解析](../mds/configParser.md)。
> 详细注册[creators.py](../entity/creators.py)