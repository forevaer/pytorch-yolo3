from entity.enum import OPTIMIZER, PHASE

# 主体设置

phase = PHASE.TRAIN
fewTrain = True
initFromWeight = False
logRegister = False
logModel = False

# 初始化文件
model_define_path = '../config/yolov3.cfg'
model_weights = '../config/yolov3.weights'
model_save_path = '../pts/model.pth'

# 训练控制
epochs = 300
batch_size = 1
image_size = 416
model_save_interval = 2
padValue = 0
nms_threshold = 0.3
confidence_threshold = 0.9
align_flip = True
multi_scale_train = True
multi_scale_interval = 10
normalized_labels = True
optimizer = OPTIMIZER.ADAM

# 数据文件
train_data = '../data/coco/trainvalno5k.txt' if not fewTrain else '../data/few/train.txt'

valid_data = '../data/coco/5k.txt' if not fewTrain else '../data/few/valid.txt'
classes_path = "../data/coco.names" if not fewTrain else '../data/few/classes.names'
detect_dir = '../data/samples' if not fewTrain else '../data/few/images'
output_path = '../output'

# weights
weights_x = 1
weights_y = 1
weights_w = 1
weights_h = 1
weights_class = 1
weights_foreground = 1
weights_background = 100





