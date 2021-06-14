
import numpy as np
from .segmentation_inspection import SegmentationInspection
from .xrai import GetMaskWithDetails


class GuidedSegmentationMapping(SegmentationInspection):

    def __init__(self):
        super(GuidedSegmentationMapping, self).__init__()
        self.caption = "Guided Segmentation Mapping"
        self.image_filters = ["color_mapping"]

    # pylint: disable=unused-argument
    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        if not target_layer_index:
            return 'Please select a layer to perform this inspection', None
        activation_maps, index, model, _, segmented_map_one_hot = self.get_segmentation_map(input_image, prediction,
                                                                                            target_layer_index)
        if index >= len(activation_maps):
            return 'Please select a class to perform this inspection', None
        image, attribution_map = self.get_attribution_map(input_image, model, activation_maps[index])
        _map = GetMaskWithDetails(image, segments=segmented_map_one_hot, base_attribution=attribution_map)[0]
        return 'success', np.expand_dims(_map, axis=0)
