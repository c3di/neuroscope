from .context_example import ContextExample
import numpy as np


class KerasSegmentationExample(ContextExample):
    def __init__(self, parent, document):
        super(KerasSegmentationExample, self).__init__(parent, document)
        self.setText("vgg_unet11 for Segmentation")
        self.setToolTip("vgg_unet11 Keras model for segmentation")

        self.backend = "Keras"
        self.model_path = "example/segmentation/vgg_unet11.h5"
        self.image_paths = ["example/segmentation/segmentation.png"]
        self.ground_true_images = ["example/segmentation/ground_true.png"]

        self.context = "Segmentation"
        self.input_shape = np.array([3, 416, 608])
        self.preprocess_setting_name = "imagenet_preset"
        self.input_image_is_rgb = False
        self.mapping_file_path = "example/segmentation/camVid_class_index.json"
        self.output_shape = np.array([12, 208, 304])
        self.output_is_whc = True
        self.add_output_activation = True
