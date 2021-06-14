# -*-coding:utf-8-*-

from PySide2.QtCore import QObject
from PySide2 import QtCore


class MVCModel(QObject):
    """
        document contain model
    """
    changed = QtCore.Signal()

    def __init__(self):
        super(MVCModel, self).__init__()
        self.model = None
        self.model_name = None
        self.images = None

    def get_model(self):
        return self.model


MVCMODEL = MVCModel()
