# -*-coding:utf-8-*-
import copy
from PIL import Image
import numpy as np
from controller.preprocessing import MODEL_PREPROCESS
from model.core import Network

CONTEXTS_PRESETS = list(['Classification', 'Segmentation'])


def is_segmentation(index):
    return index == 1

#pylint:  disable=too-few-public-methods
class Model(Network):
    """
        model base class containing optimizer
    """
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Model'
        self.native_model = None
        self.preprocess_setting_name = 'imagenet_preset'
        self.preprocess_setting = {'mean': [103.939, 116.779, 123.68], 'std': [0.0, 0.0, 0.0]}
        self.context = 'Segmentation'
        self.input_shape = np.array([3, 416, 608])
        self.output_shape = np.array([2, 208, 304])
        self.should_normalize = False
        self.channel_first = False
        self.image_is_rgb = True
        self.output_is_whc = True
        self.add_output_activation = True
        self.mapping_file_path = ""

    def set_context(self, index):
        self.context = CONTEXTS_PRESETS[index]

    def update_preprocess_setting(self, preset_name):
        self.preprocess_setting = MODEL_PREPROCESS.get_preset_values(preset_name)

    def resize_input(self, img, width=None, height=None):
        if width is None:
            width = self.native_model.input_shape[2]
        if height is None:
            height = self.native_model.input_shape[1]
        img = Image.fromarray(img)
        img = img.resize((width, height))
        img = np.array(img)
        return img

    def deepcopy(self):
        copyed_model = copy.copy(self)
        copyed_model.native_model = copy.deepcopy(self.native_model)
        return copyed_model
