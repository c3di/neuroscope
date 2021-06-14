# -*-coding:utf-8-*-
import os
import json
import tempfile
import h5py
import numpy as np
import keras
import keras.backend as K
from tensorflow.python.framework import ops
import tensorflow as tf
from model.core import Model, Layer
from controller.backends.backend import Backend
from .h5dict import H5Dict


class KerasBackend(Backend):
    def load_architecture(self, file_path):
        file_reader = open(file_path, 'r')
        loaded_model_json = file_reader.read()
        file_reader.close()
        config = json.loads(loaded_model_json)
        model = self._keras_model_from_json(config['config'])
        model.native_model = keras.models.model_from_json(loaded_model_json)
        return model

    def load_model(self, file_path):
        model = self._keras_model_from_hdf5(file_path)
        model.native_model = keras.models.load_model(file_path)
        return model

    def predict(self, model, model_input):
        K.set_learning_phase(0)
        return model.native_model.predict(model_input)

    def _keras_model_from_json(self, network_config):
        model = Model()
        model.network_config = network_config
        model.name = network_config['name']
        index = 0
        for layer_config in network_config['layers']:
            layer = Layer(layer_config["config"]['name'], layer_config['class_name'],
                          layer_config['config'], None, index)
            layer.inbound_layers = layer_config['inbound_nodes']
            model.layers_append(layer)
            index = index + 1
        for layer in model.layers:
            all_inbound_layers = []
            if layer.inbound_layers:
                for inbound_layer_node in layer.inbound_layers[0]:
                    inbound_name = inbound_layer_node[0]
                    inbound_layer = model.get_layer_by_name(inbound_name)
                    all_inbound_layers.append(inbound_layer)
            layer.inbound_layers = all_inbound_layers
        model.input_layer = model.get_layer_by_name(network_config['input_layers'][0][0])
        return model

    def _keras_model_from_hdf5(self, filepath):
        """load keras mode from json file"""
        h5dict = H5Dict(filepath, mode='r')
        try:
            """ deserialize mode from h5dict"""
            model_config = h5dict['model_config']
            if model_config is None:
                raise ValueError('No model found in config.')
            model_config = model_config.decode('utf-8')
            config = json.loads(model_config)
            model = self._keras_model_from_json(config['config'])
        finally:
            if not isinstance(filepath, h5py.Group):
                h5dict.close()
        return model

    def forward_pass_up_to_layer(self, model, model_input, target_layer_index):
        # here the input only support minibatch.(batchsize, channel, width, height")
        K.set_learning_phase(0)
        target_layer_output = model.native_model.layers[target_layer_index].output
        if model.native_model.input != target_layer_output:
            compute_output_function = K.function(inputs=[model.native_model.input], outputs=[target_layer_output])
            output = compute_output_function([model_input])[0]
            output = np.squeeze(output)
            if output.ndim == 3:
                output = output.transpose(2, 0, 1)
        else:
            output = model_input
        return output

    def gradients_of_output_wrt_input(self, model, model_input, model_input_layer_index, class_index):
        K.set_learning_phase(0)
        desire_layer = model.native_model.layers[model_input_layer_index].output
        gradients = K.gradients(model.native_model.output[0][..., class_index], desire_layer)
        compute_gradients = K.function(inputs=[model.native_model.input], outputs=gradients)
        gradients = compute_gradients([model_input])
        return gradients[0]

    def _register_guided_gradient(self, name):
        # pylint: disable=protected-access
        if name not in ops._gradient_registry._registry:
            @tf.RegisterGradient(name)
            def _guided_backpro(layer_op, grad):
                dtype = layer_op.outputs[0].dtype
                gate_g = tf.cast(grad > 0., dtype)
                gate_y = tf.cast(layer_op.outputs[0] > 0., dtype)
                return gate_y * gate_g * grad

    def register_guided_relu_gradients(self, model):
        # pylint: disable=protected-access
        tempfile_name = tempfile._get_candidate_names()
        tempfile_name = next(tempfile_name) + '.h5'
        model_path = os.path.join(tempfile.gettempdir(), tempfile_name)
        try:
            model.native_model.save(model_path)
            # register modifier and load modified model under custom context.
            self._register_guided_gradient('guided_relu')
            # create graph under custom context manager
            with tf.get_default_graph().gradient_override_map({'Relu': 'guided_relu'}):
                # this should rebuild graph with modifications.
                model.native_model = keras.models.load_model(model_path)
        finally:
            os.remove(model_path)
        return model
