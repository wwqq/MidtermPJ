from .YoloV2V3.Model import Model as Yolo2
from .YoloV2V3.Model import Model as Yolo3
from .Faster_RCNN.Model import Model as Faster_RCNN

models = {
    'Yolo2': Yolo2,
    'Yolo3': Yolo3,
    'Faster_RCNN': Faster_RCNN,
    'FRCNN': Faster_RCNN,
}

def get_model(model: str):
    if model is None:
        raise AttributeError('--model MUST be specified now, available: {%s}.' % ('|'.join(models.keys())))

    if model in models:
        return models[model]
    else:
        raise AttributeError('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

