# -*-coding:utf-8-*-

import numpy as np
from scipy.ndimage.interpolation import zoom
from model.MVCModel import MVCMODEL
from controller.backends.backend_provider import BACKEND_PROVIDER
from .inspection import Inspection


class GradCamInspection(Inspection):
    """
        gradient weighted class activation map plus plus
        reference: Grad-CAM: Visual Explanations from Deep
        Networks via Gradient-based Localization
        https://arxiv.org/abs/1610.02391
    """

    def __init__(self):
        super(GradCamInspection, self).__init__()
        self.caption = "Grad Cam"
        self.image_filters = ["color_mapping", "heatmap",
                              "heatmap_pos", "heatmap_neg"]

    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        if target_layer_index is None:
            return 'Please select a layer to perform this inspection', None
        current_activated_backend = BACKEND_PROVIDER.get_current_backend()
        model = MVCMODEL.get_model()
        input_image = self.input_preprocessing(model, input_image[0])
        target_layer_output = current_activated_backend.forward_pass_up_to_layer(model, input_image, target_layer_index)
        weights = current_activated_backend.gradients_of_output_wrt_input(model, input_image, target_layer_index,
                                                                          prediction.index)
        weights = weights[0]
        weights = np.mean(weights, (0, 1))
        target_layer_output = target_layer_output.transpose(1, 2, 0)
        weighted_output = weights * target_layer_output
        weighted_output = np.sum(weighted_output, (-1))
        weighted_output = np.maximum(weighted_output, 0)
        zoomed_weighted_output = zoom(weighted_output, (model.input_shape[1] / weighted_output.shape[0],
                                                        model.input_shape[2] / weighted_output.shape[1]))
        return 'success', np.expand_dims(zoomed_weighted_output, axis=0)
