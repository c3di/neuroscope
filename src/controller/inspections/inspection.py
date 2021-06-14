# -*-coding:utf-8-*-

from PySide2.QtCore import QObject
from controller.preprocessing import MODEL_PREPROCESS


class Inspection(QObject):
    """
        Inspection base class
    """
    def __init__(self):
        super(Inspection, self).__init__()
        self.image_filters = None
        self.caption = None

    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        raise NotImplementedError

    @property
    def image_filter(self):
        return self.image_filter

    def input_preprocessing(self, model, input_image):
        if model is None:
            return input_image
        if model.preprocess_setting is None:
            raise Exception("Please set mean and standard deviation for input preprocessing")
        input_data = model.resize_input(input_image, width=model.input_shape[2], height=model.input_shape[1])
        return MODEL_PREPROCESS(input_data[..., :3], model.preprocess_setting['mean'],
                                model.preprocess_setting['std'], model.should_normalize, model.channel_first,
                                model.image_is_rgb)
