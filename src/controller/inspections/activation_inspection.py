# -*-coding:utf-8-*-

import numpy as np
from model.MVCModel import MVCMODEL
from controller.backends.backend_provider import BACKEND_PROVIDER
from .inspection import Inspection


class ActivationInspection(Inspection):
    """
    activation feature maps in the deep learning model for exactly layer
    """

    def __init__(self):
        super(ActivationInspection, self).__init__()
        self.caption = "Activation maps"
        self.image_filters = ["none", "color_mapping", "color_mapping_on_each_image", "heatmap"]

    # pylint: disable= unused-argument
    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        if target_layer_index == 0:
            return 'success', np.expand_dims(input_image, axis=0)
        if target_layer_index is None:
            return 'Please select a layer to perform this inspection', None
        current_activated_backend = BACKEND_PROVIDER.get_current_backend()
        model = MVCMODEL.get_model()
        input_image = self.input_preprocessing(model, input_image[0])
        activation_maps = current_activated_backend.forward_pass_up_to_layer(model, input_image, target_layer_index)
        return 'success', activation_maps
