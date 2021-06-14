import os
import numpy as np
import imageio
from PySide2.QtWidgets import QAction
from model.MVCModel import MVCMODEL
from controller.backends import BACKEND_PROVIDER


class ContextExample(QAction):
    def __init__(self, parent, document, ):
        super(ContextExample, self).__init__()
        self.main_window = parent
        self.document = document
        self.setText("Context Example")
        self.setToolTip("")

        self.backend = ""
        self.model_path = ""
        self.image_paths = []
        self.ground_true_images = []

        self.context = ""
        self.input_shape = np.array([0, 0, 0])
        self.preprocess_setting_name = ""
        self.input_image_normalization = False
        self.input_image_is_channel_first = False
        self.input_image_is_rgb = False
        self.mapping_file_path = ""
        self.output_shape = np.array([0, 0, 0])
        self.output_is_whc = True
        self.add_output_activation = True
        self.triggered.connect(self.perform_action)

    def perform_action(self):
        model = self.perform_model_loading()
        self.main_window.main_area.closeAllSubWindows()
        self.main_window.open_architecture_window()
        self.perform_model_option_setting(model)
        self.main_window.open_image_input_window()
        self.main_window.input_images.remove_all_image()
        self.perform_image_import()
        self.main_window.add_inspection_window()
        self.main_window.fit_network_to_window_size.trigger()

    def perform_image_import(self):
        image_idx = []
        for path in self.image_paths:
            image = imageio.imread(path)[..., :3]
            file_name = os.path.basename(path)
            idx = self.main_window.input_images.add_image(image, file_name)
            image_idx.append(idx - 1)
        for index, path in enumerate(self.ground_true_images):
            image = imageio.imread(path)
            file_name = os.path.basename(path)
            self.main_window.input_images.add_ground_truth(image_idx[index], image, file_name)

    def perform_model_option_setting(self, model):
        model.context = self.context
        model.input_shape = self.input_shape
        model.preprocess_setting_name = self.preprocess_setting_name
        model.should_normalize = self.input_image_normalization
        model.channel_first = self.input_image_is_channel_first
        model.image_is_rgb = self.input_image_is_rgb
        model.mapping_file_path = self.mapping_file_path
        model.output_shape = self.output_shape
        model.output_is_whc = self.output_is_whc
        model.add_output_activation = self.add_output_activation

    def perform_model_loading(self):
        BACKEND_PROVIDER.activate_backend(self.backend)
        backend = BACKEND_PROVIDER.get_current_backend()
        model = backend.load(self.model_path, "model")
        MVCMODEL.model = model
        MVCMODEL.model_name = os.path.basename(self.model_path)
        MVCMODEL.changed.emit()
        return model
