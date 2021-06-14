# -*-coding:utf-8-*-


class Network():
    """
        directed acyclic graph
    """
    def __init__(self):

        self._layers = []
        self.name = 'Network'
        self._input_layer = None

    @property
    def input_layer(self):
        return self._input_layer

    @input_layer.setter
    def input_layer(self, new_input_layer):
        self._input_layer = new_input_layer

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, layers):
        self._layers = layers

    def get_layer_by_name(self, layer_name):
        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        return None

    def get_layer_by_index(self, layer_index):
        for layer in self.layers:
            if layer.index == layer_index:
                return layer
        return None

    def layers_append(self, *layers):
        for layer in layers:
            self._layers.append(layer)
