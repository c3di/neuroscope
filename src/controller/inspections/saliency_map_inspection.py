# -*-coding:utf-8-*-

import numpy as np
from model.MVCModel import MVCMODEL
from controller.backends.backend_provider import BACKEND_PROVIDER
from .inspection import Inspection


class SaliencyMapInspection(Inspection):
    """
       Deep Inside Convolutional Networks: Visualising Image Classification
       Models and Saliency Maps
       reference: https://arxiv.org/abs/1312.6034
    """
    def __init__(self):
        super(SaliencyMapInspection, self).__init__()
        self.caption = "Saliency Map"
        self.image_filters = ["normalized", "color_mapping", "pos_norm", "neg_norm"]

    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        current_activated_backend = BACKEND_PROVIDER.get_current_backend()
        model = MVCMODEL.get_model()
        input_image = self.input_preprocessing(model, input_image[0])
        saliency_maps = current_activated_backend.gradients_of_output_wrt_input(model, input_image, 0, prediction.index)
        saliency_map = np.max(saliency_maps, axis=3)
        return 'success', saliency_map
