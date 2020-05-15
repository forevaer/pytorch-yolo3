from entity.enum import OPTIMIZER, PHASE

# 主体设置

phase = PHASE.DETECT
mini = True
initFromWeight = False
logRegister = False
logModel = False
align_flip = True
multi_scale_train = True
multi_scale_interval = 10
view = True

# 初始化文件
model_define_path = '../config/yolov3.cfg'
model_weights = '../config/yolov3.weights'
model_save_path = '../pts/model.pth'

# 训练控制
epochs = 300
batch_size = 10
image_size = 416
model_save_interval = 2
padValue = 0
nms_threshold = 0.3
confidence_threshold = 0.9
normalized_labels = True
optimizer = OPTIMIZER.ADAM
viewer_limit = 10

# 数据文件
mini_limit = 1
train_data = '../data/coco/train.txt'
valid_data = '../data/coco/valid.txt'
classes_path = "../data/coco.names"
detect_dir = '../data/detect'
output_path = '../output'

# weights
weights_x = 1
weights_y = 1
weights_w = 1
weights_h = 1
weights_class = 1
weights_foreground = 1
weights_background = 100





