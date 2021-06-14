# -*-coding:utf-8-*-

from .inspection import Inspection
from .guided_back_prop_inspection import GuidedBackPropInspection
from .grad_cam_inspection import GradCamInspection


class GuidedGradCamInspection(Inspection):
    """
        gradient weighted class activation map plus plus
        reference: Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization
        https://arxiv.org/abs/1610.02391
    """
    def __init__(self):
        super(GuidedGradCamInspection, self).__init__()
        self.caption = "Guided Grad Cam"
        self.image_filters = ["normalized", "color_mapping", "pos_norm", "neg_norm"]

    # pylint: disable=unused-argument
    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        message, saliency_map = GuidedBackPropInspection().perform(input_image, target_layer_index, prediction)
        if saliency_map is None:
            return message, None
        message, cam = GradCamInspection().perform(input_image, target_layer_index, prediction)
        if cam is None:
            return message, None
        weighted_saliency_map = saliency_map * cam
        return 'success', weighted_saliency_map
