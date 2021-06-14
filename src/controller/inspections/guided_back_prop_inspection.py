# -*-coding:utf-8-*-

import numpy as np
from model.MVCModel import MVCMODEL
from controller.backends.backend_provider import BACKEND_PROVIDER
from .inspection import Inspection

class GuidedBackPropInspection(Inspection):
    """
        guided back propagation
        reference: Striving for Simplicity: The All Convolutional Net
        https://arxiv.org/abs/1412.6806
    """
    def __init__(self):
        super(GuidedBackPropInspection, self).__init__()
        self.caption = "Guided Back Prop"
        self.image_filters = ["normalized", "color_mapping" ,"pos_norm", "neg_norm"]

    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        current_activated_backend = BACKEND_PROVIDER.get_current_backend()
        model = MVCMODEL.get_model()
        input_image = self.input_preprocessing(model, input_image[0])
        model = model.deepcopy()
        model = current_activated_backend.register_guided_relu_gradients(model)
        saliency_maps = current_activated_backend.gradients_of_output_wrt_input(model, input_image, 0, prediction.index)
        saliency_map = np.max(saliency_maps, axis=3)
        return 'success', saliency_map
