# -*-coding:utf-8-*-

from PySide2.QtCore import QObject
from .input_inspection import InputInspection
from .activation_inspection import ActivationInspection
from .saliency_map_inspection import SaliencyMapInspection
from .guided_back_prop_inspection import GuidedBackPropInspection
from .grad_cam_inspection import GradCamInspection
from .guided_grad_cam_inspection import GuidedGradCamInspection
from .grad_cam_plus_inspection import GradCamPlusInspection
from .segmentation import Segmentation
from .confusion_matrix import Confusion_matrix
from .guided_segmentation_mapping import GuidedSegmentationMapping
from .segmented_score_map import SegmentedScoreMap
from .similarity_map import SimilarityMap
from .fusion_map import FusionMap


class InspectionRegistry(QObject):
    """
        inspections management
    """

    def __init__(self):
        super(InspectionRegistry, self).__init__()
        self.available_inspections = dict()
        self.register_inspection()

    def register_inspection(self):
        input_inspection = InputInspection()
        activation_inspection = ActivationInspection()
        saliency_inspection = SaliencyMapInspection()
        guided_backprop = GuidedBackPropInspection()
        grad_cam = GradCamInspection()
        guided_cam = GuidedGradCamInspection()
        grad_cam_plus = GradCamPlusInspection()
        segmentation = Segmentation()
        confusion_matrix = Confusion_matrix()
        gssm = GuidedSegmentationMapping()
        ssm = SegmentedScoreMap()
        similarity = SimilarityMap()
        fusion_map = FusionMap()
        self.available_inspections['Classification'] = [input_inspection,
                                                        activation_inspection,
                                                        saliency_inspection,
                                                        guided_backprop,
                                                        grad_cam,
                                                        guided_cam,
                                                        grad_cam_plus]
        self.available_inspections['Segmentation'] = [input_inspection,
                                                      activation_inspection,
                                                      saliency_inspection,
                                                      guided_backprop,
                                                      grad_cam,
                                                      guided_cam,
                                                      ssm,
                                                      gssm,
                                                      similarity,
                                                      segmentation,
                                                      fusion_map,
                                                      confusion_matrix]

    def all_inspection_names(self, context='Classification'):
        names = []
        if context in self.available_inspections:
            for inspection in self.available_inspections[context]:
                names.append(inspection.caption)
        return names

    def get_by_index(self, index, context='Classification'):
        if context not in self.available_inspections:
            return None
        return self.available_inspections[context][index]
