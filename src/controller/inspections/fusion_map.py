import numpy as np
from .inspection import Inspection
from .predict_inspection import PredictInspection


class FusionMap(Inspection):
    """
        Fusion segmentation map
    """

    def __init__(self):
        super(FusionMap, self).__init__()
        self.caption = "Fusion Map"
        self.image_filters = ["none", "color_mapping", "heatmap"]

    # pylint: disable=unused-argument
    def perform(self, input_image, layer=None, prediction=None, settings=None):
        image = input_image
        _, pred = PredictInspection().perform(image)
        pred_map = np.full(pred[0].result.shape, -np.inf)
        for i in range(0, len(pred) - 2):
            if settings.selected_classes is not None:
                if settings.selected_classes[i]:
                    pred_map = np.dstack((pred_map, pred[i].result))
            else:
                pred_map = np.dstack((pred_map, pred[i].result))
        fusion_map = np.max(pred_map, -1)
        fusion_map = fusion_map * 255
        result = np.expand_dims(fusion_map, axis=0).astype(np.uint8)
        return 'success', result
