from .context_example import ContextExample
import numpy as np


class KerasSegmentationCognitwinExample(ContextExample):
    def __init__(self, parent, document):
        super(KerasSegmentationCognitwinExample, self).__init__(parent, document)
        self.setText("mini-unet for Cognitwin")
        self.setToolTip("mini-unet Keras model for segmentation")

        self.backend = "Keras"
        self.model_path = "example/segmentation/mini-unet_model-COGNITWIN-Keras.h5"
        self.image_paths = ["example/segmentation/20201125_camera_2_0000.png"]

        self.context = "Segmentation"
        self.input_shape = np.array([3, 544, 960])
        self.preprocess_setting_name = "imagenet_preset"
        self.input_image_is_rgb = False
        self.mapping_file_path = "example/segmentation/unet_model_COGNITWIN_Keras.json"
        self.output_shape = np.array([2, 544, 960])
        self.output_is_whc = True
        self.add_output_activation = True
