import cv2
import numpy as np
from .segmentation_inspection import SegmentationInspection
from .xrai import GetMaskWithDetails


class SimilarityMap(SegmentationInspection):

    def __init__(self):
        super(SimilarityMap, self).__init__()
        self.caption = "Similarity map"
        self.image_filters = ["color_mapping"]

    # pylint: disable=unused-argument
    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        if target_layer_index is None:
            return 'Please select a layer to perform this inspection', None
        activation_maps, index, _, segmented_map, segmented_map_one_hot = self.get_segmentation_map(input_image,
                                                                                                    prediction,
                                                                                                    target_layer_index)
        if settings.selected_classes is not None:
            class_map = settings.selected_classes
            class_map[index] = True
            segmented_map_one_hot = segmented_map_one_hot[class_map, :, :]

        image = cv2.resize(input_image[0], dsize=(segmented_map.shape[1], segmented_map.shape[0]),
                           interpolation=cv2.INTER_CUBIC)
        if index >= len(activation_maps):
            return 'Please select a class to perform this inspection', None
        attribution_map = activation_maps[index]
        attribution_map = cv2.resize(attribution_map, dsize=(image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)
        attribution_map = np.stack((attribution_map, attribution_map, attribution_map), -1)
        _map = GetMaskWithDetails(image, segments=segmented_map_one_hot, base_attribution=attribution_map,
                                  just_output_maps=True)[0]
        return 'success', np.expand_dims(_map, axis=0)
