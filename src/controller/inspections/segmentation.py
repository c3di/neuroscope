import numpy as np
from .inspection import Inspection


class Segmentation(Inspection):
    """
        segmented image visualization
    """

    def __init__(self):
        super(Segmentation, self).__init__()
        self.caption = "Segmented image"
        self.image_filters = ["none", "color_mapping", "heatmap",
                              "heatmap_pos", "ground_truth", "intersection", "union", "error"]

    # pylint: disable=unused-argument
    def perform(self, input_image, layer=None, prediction=None, settings=None):
        image = prediction.result * 255
        result = np.expand_dims(image, axis=0).astype(np.uint8)
        return 'success', result
