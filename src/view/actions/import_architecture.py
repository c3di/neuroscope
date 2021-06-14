import os
from PySide2.QtWidgets import QAction, QFileDialog, QMessageBox
from PySide2.QtGui import QIcon
from PySide2.QtCore import Qt
from model.MVCModel import MVCMODEL
from controller.backends import BACKEND_PROVIDER


class ImportArchitecture(QAction):
    def __init__(self, parent, document):

        super(ImportArchitecture, self).__init__(QIcon('resources/icons/layers.svg'),
                                                 'Import &Network', parent)
        self.setToolTip("Import network file")
        self.triggered.connect(self.perform_action)
        self.main_window = parent
        self.document = document

    def perform_action(self):
        file_path = QFileDialog.getOpenFileName(self.main_window,
                                                "Load Network Architecture",
                                                filter="All files(*.*);;"
                                                       "JSON files(*.json);; H5DF(*.h5)")
        if file_path[0]:
            self.main_window.open_architecture_window()
            self.main_window.app.setOverrideCursor(Qt.WaitCursor)
            model_name, extension = os.path.splitext(file_path[0])
            if extension == '.h5':
                BACKEND_PROVIDER.activate_backend('Keras')
            elif extension == '.pt':
                BACKEND_PROVIDER.activate_backend('Pytorch')
            else:
                self.main_window.app.restoreOverrideCursor()
                QMessageBox.critical(self.main_window,
                                     "The selected file is not supported",
                                     "The file is corrupted or does not contain a supported model",
                                     QMessageBox.Ok)
                return
            backend = BACKEND_PROVIDER.get_current_backend()
            model = backend.load(file_path[0], 'model')
            MVCMODEL.model = model
            MVCMODEL.model_name = os.path.basename(file_path[0])
            MVCMODEL.changed.emit()
            self.main_window.main_area.closeAllSubWindows()
            self.main_window.app.restoreOverrideCursor()
