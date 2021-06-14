# -*-coding:utf-8-*-

import cv2
import numpy as np
from model.MVCModel import MVCMODEL
from controller.backends.backend_provider import BACKEND_PROVIDER
from .inspection import Inspection
from .predict_inspection import PredictInspection


class SegmentationInspection(Inspection):
    """
        Segmentation Inspection base class
    """

    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        raise NotImplementedError

    def get_segmentation_map(self, input_image, prediction, target_layer_index):
        index = prediction.index
        _, pred = PredictInspection().perform(input_image)
        pred = pred[-1]
        current_activated_backend = BACKEND_PROVIDER.get_current_backend()
        model = MVCMODEL.get_model()
        preprocessed_image = self.input_preprocessing(model, input_image[0])
        output_classes = model.output_shape[0]
        activation_maps = current_activated_backend.forward_pass_up_to_layer(model, preprocessed_image,
                                                                             target_layer_index)
        segmented_map = pred.result * output_classes
        segmented_map_one_hot = np.eye(output_classes)[segmented_map.astype('uint8')]
        segmented_map_one_hot = np.rollaxis(segmented_map_one_hot, 2)
        return activation_maps, index, model, segmented_map, segmented_map_one_hot

    def get_attribution_map(self, input_image, model, attribution_map):
        image = cv2.resize(input_image[0], dsize=(model.output_shape[2], model.output_shape[1]),
                           interpolation=cv2.INTER_CUBIC)
        attribution_map = cv2.resize(attribution_map, dsize=(image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)
        attribution_map = np.stack((attribution_map, attribution_map, attribution_map), -1)
        return image, attribution_map
