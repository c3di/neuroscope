import numpy as np
from model.MVCModel import MVCMODEL
from controller.backends.backend_provider import BACKEND_PROVIDER
from .segmentation_inspection import SegmentationInspection
from .xrai import GetMaskWithDetails


class SegmentedScoreMap(SegmentationInspection):

    def __init__(self):
        super(SegmentedScoreMap, self).__init__()
        self.caption = "Segmented Score Map"
        self.image_filters = ["color_mapping"]

    # pylint: disable=unused-argument
    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        if target_layer_index is None:
            return 'Please select a layer to perform this inspection', None
        index = prediction.index
        current_activated_backend = BACKEND_PROVIDER.get_current_backend()
        model = MVCMODEL.get_model()
        preprocessed_image = self.input_preprocessing(model, input_image[0])
        activation_maps = current_activated_backend.forward_pass_up_to_layer(model, preprocessed_image,
                                                                             target_layer_index)
        if index >= len(activation_maps):
            return 'Please select a class to perform this inspection', None
        image, attribution_map = self.get_attribution_map(input_image, model, activation_maps[index])
        _map = GetMaskWithDetails(image, segments=None, base_attribution=attribution_map)[0]
        return 'success', np.expand_dims(_map, axis=0)
