# -*-coding:utf-8-*-

import numpy as np
from .inspection import Inspection


class InputInspection(Inspection):
    """
        input image visualization
    """

    def __init__(self):
        super(InputInspection, self).__init__()
        self.caption = "Input Image"
        self.image_filters = ["none"]

    # pylint: disable=unused-argument
    def perform(self, input_image, layer=None, prediction=None, settings=None):
        return 'success', np.expand_dims(input_image[0], axis=0)
