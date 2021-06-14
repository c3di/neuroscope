# -*-coding:utf-8-*-
from enum import Enum


class Domain_Layer_Types(Enum):
    Default = 'Default'
    InputLayer = 'InputLayer'
    Conv2D = 'Conv2D'
    MaxPooling2D = 'MaxPooling2D'
    AveragePooling2D = 'AveragePooling2D'
    Flatten = 'Flatten'
    Dense = 'Dense'
    BatchNormalization = 'BatchNormalization'
    Activation = 'Activation'
    Add = 'Add'
    Concatenate = 'Concatenate'
    Reshape = 'Reshape'
    ReLu = 'Relu'
    Dropout = 'Dropout'
    upsample_bilinear2d = 'Upsample_bilinear'
    Cat = 'Cat'
    View = 'View'
    Sigmoid = 'Sigmoid'

    @classmethod
    def has_value(cls, value):
        """check whether value belong to the enum or not"""
        return any(value == item.value for item in cls)

class Layer:
    """ layer base class"""

    def __init__(self, name='layer', layer_class='Layer', config=None, shape_of_output=None, index=None):
        super(Layer, self).__init__()
        self._config = config
        self._inbound_layers = []
        self.name = name
        self.index = index
        self._layer_class = layer_class
        self.shape_of_output = shape_of_output
        self.scope_name = None  # for Pytorch
    @property
    def layer_class(self):
        return self._layer_class

    @layer_class.setter
    def layer_class(self, new_class):
        self._layer_class = new_class

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, layer_config):
        self._config = layer_config

    @property
    def inbound_layers(self):
        return self._inbound_layers

    @inbound_layers.setter
    def inbound_layers(self, inbound_layers):
        self._inbound_layers = inbound_layers
