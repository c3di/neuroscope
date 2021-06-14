# -*-coding:utf-8-*-
from abc import ABC


class Backend(ABC):
    """the base class of backend"""
    def load(self, file_path, file_type):
        """Current we support network architecture, weights
         and the whole network(architecture and parameters) import"""
        if file_type == 'architecture':
            return self.load_architecture(file_path)
        elif file_type == 'model':
            return self.load_model(file_path)
        else:
            raise Exception("illegal file_type")

    def load_architecture(self, file_path):
        raise NotImplementedError

    def load_model(self, file_path):
        raise NotImplementedError

    def predict(self, model, model_input):
        raise NotImplementedError

    def forward_pass_up_to_layer(self, model, model_input, target_layer_index):
        raise NotImplementedError

    def gradients_of_output_wrt_input(self, model, model_input, model_input_layer_index, class_index):
        raise NotImplementedError

    def register_guided_relu_gradients(self, model):
        raise NotImplementedError
