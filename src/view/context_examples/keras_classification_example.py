from .context_example import ContextExample
import numpy as np


class KerasClassificationExample(ContextExample):
    def __init__(self, parent, document):
        super(KerasClassificationExample, self).__init__(parent, document)
        self.setText("VGG16 for Classification")
        self.setToolTip("VGG16 Keras model for classification")

        self.backend = "Keras"
        self.model_path = "example/classification/Vgg16.h5"
        self.image_paths = ["example/classification/cat_dog.png",
                            "example/classification/water-bird.png"]

        self.context = "Classification"
        self.input_shape = np.array([3, 224, 224])
        self.preprocess_setting_name = "imagenet_preset"
        self.input_image_is_rgb = False
        self.mapping_file_path = "example/classification/imagenet_class_index.json"
