# -*-coding:utf-8-*-

import numpy as np
from scipy.ndimage.interpolation import zoom
from controller.inspections import Inspection
from controller.backends import BACKEND_PROVIDER
from model.MVCModel import MVCMODEL


class GradCamPlusInspection(Inspection):
    """
        gradient weighted class activation map plus plus
        reference: Grad-CAM++: Improved Visual Explanations for
        Deep Convolutional Networks
        https://arxiv.org/abs/1710.11063
    """

    def __init__(self):
        super(GradCamPlusInspection, self).__init__()
        self.caption = "Grad Cam Plus"
        self.image_filters = ["color_mapping", "heatmap",
                              "heatmap_pos", "heatmap_neg"]

    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        if target_layer_index is None:
            return 'Please select a layer to perform this inspection', None
        current_activated_backend = BACKEND_PROVIDER.get_current_backend()
        model = MVCMODEL.get_model()
        input_image = self.input_preprocessing(model, input_image[0])
        class_specified_output = current_activated_backend.predict(model, input_image)[..., prediction.index]
        target_layer_output = current_activated_backend.forward_pass_up_to_layer(model, input_image, target_layer_index)
        target_layer_output = target_layer_output.transpose(1, 2, 0)
        gradients = current_activated_backend.gradients_of_output_wrt_input(model, input_image, target_layer_index,
                                                                            prediction.index)
        weights_k = self.get_weight(class_specified_output, gradients, target_layer_output)
        weighted_output = np.sum(weights_k * target_layer_output, axis=2)
        zoomed_weighted_output = zoom(weighted_output, (model.input_shape[1] / weighted_output.shape[0],
                                                        model.input_shape[2] / weighted_output.shape[1]))
        return 'success', np.expand_dims(zoomed_weighted_output, axis=0)

    def get_weight(self, class_specified_output, gradients, target_layer_output):
        # https://arxiv.org/pdf/1710.11063.pdf
        first = np.exp(class_specified_output) * gradients[0]
        second = first * gradients[0]
        third = second * gradients[0]
        global_sum = np.sum(target_layer_output, axis=(0, 1))
        alpha_denom = 2.0 * second + global_sum * third
        alpha_denom[alpha_denom == 0] = 1.0
        alpha = second / alpha_denom
        weights = np.maximum(first[0], 0.0)
        alpha_sum = np.sum(alpha, axis=(0, 1))
        alpha_sum[alpha_sum == 0] = 1.0
        alpha = alpha / alpha_sum
        weights_k = np.sum(weights * alpha, axis=(0, 1))
        return weights_k
