MODEL:
  NAME: Yolo2
  LOAD: ./pretrained/darknet19_448.conv.23
DATA:
  DATASET: voc
  TRANSFORM: yolo2
  SCALE: 448
OPTIMIZE:
  OPTIMIZER: sgd
  BASE_LR: 0.0004
  SCHEDULER: 10x
  BATCH_SIZE: 24
TEST:
  NMS_THRESH : 0.5  # nms iou thresh at test time
  CONF_THRESH: 0.05 # confidence thresh to keep at test time
MISC:
  VAL_FREQ: 5
  SAVE_FREQ: 5
  NUM_WORKERS: 2
