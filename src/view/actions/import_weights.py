from PySide2.QtCore import Slot
from PySide2.QtWidgets import QAction, QFileDialog, QMessageBox
from PySide2.QtGui import QIcon
from controller.backends import BACKEND_PROVIDER


class ImportWeights(QAction):
    def __init__(self, parent, document):
        super(ImportWeights, self).__init__(QIcon('resources/icons/add-weights.svg'),
                                            'Import Weights', parent)
        self.setToolTip("Import network weights file")
        self.triggered.connect(self.perform_action)
        self.main_window = parent
        self.document = document

    @Slot()
    def perform_action(self):
        file_name = QFileDialog.getOpenFileName(self.main_window,
                                                "Load Network Weights",
                                                filter="HDF5 (*.h5)")[0]
        if file_name:
            try:
                backend = BACKEND_PROVIDER.get_current_backend()
            # pylint: disable=broad-except
            except Exception:
                QMessageBox.critical(self.main_window,
                                     "Backend Missing",
                                     "No deep learning backend was installed.\n"
                                     "Please check your Neuroscope installation"
                                     "to make sure at least one backend is installed.",
                                     QMessageBox.Ok)
                return
            try:
                backend.load_weights(file_name)
            # pylint: disable=broad-except
            except Exception:
                QMessageBox.critical(self.main_window,
                                     "The selected file is not supported",
                                     "The file is corrupted or does not contain supported weights",
                                     QMessageBox.Ok)
                return
