from PySide2.QtWidgets import QAction, QMessageBox, QFileDialog
from PySide2.QtGui import QIcon


class SelectGroundTruthDir(QAction):
    def __init__(self, parent, input_images):
        super(SelectGroundTruthDir, self).__init__(QIcon('resources/icons/import_image.svg'),
                                                   "Ground truth Dir", parent)
        self.setToolTip("Import Ground truth files")
        self.triggered.connect(self.perform_action)
        self.input_images = input_images
        self.main_window = parent

    def perform_action(self):
        result = QFileDialog.getExistingDirectory(self.main_window,
                                                  "select ground truth directory")
        # pylint disable=invalid-name
        if result == "":
            return
        try:
            self.input_images.set_ground_truth_directory(result)
            # pylint: disable=broad-except
        except Exception:
            QMessageBox.critical(self.main_window,
                                 "The selected directory is not supported",
                                 QMessageBox.Ok)
        self.main_window.open_image_input_window()
