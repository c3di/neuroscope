from PySide2 import QtCore
from model.core import Layer


class SignalHub(QtCore.QObject):

    layer_selected = QtCore.Signal(Layer)
    model_changed = QtCore.Signal()

    def __init__(self):
        super(SignalHub, self).__init__()

    @QtCore.Slot(Layer)
    def on_layer_selected(self, layer):
        self.layer_selected.emit(layer)

    @QtCore.Slot()
    def on_model_changed(self):
        self.model_changed.emit()
