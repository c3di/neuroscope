# -*-coding:utf-8-*-
from model.MVCModel import MVCMODEL
from controller.backends.backend_provider import BACKEND_PROVIDER
from controller.decode_prediction import DECODE_PREDICTIONS
from .import Inspection


class PredictInspection(Inspection):
    """
        image prediction results from a deep learning model
    """
    def __init__(self):
        super(PredictInspection, self).__init__()
        self.caption = "Prediction"

    #pylint: disable=unused-argument
    def perform(self, input_image, target_layer_index=None, prediction=None, settings=None):
        current_activated_backend = BACKEND_PROVIDER.get_current_backend()
        model = MVCMODEL.get_model()
        preprocessed_image = self.input_preprocessing(model, input_image[0])
        predictions = current_activated_backend.predict(model, preprocessed_image)
        decoded_results = DECODE_PREDICTIONS(model, input_image, predictions)
        return 'success', decoded_results


