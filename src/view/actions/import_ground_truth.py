import os
import imageio
from PySide2.QtWidgets import QAction, QMessageBox, QFileDialog
from PySide2.QtGui import QIcon


class ImportGroundTruth(QAction):
    def __init__(self, parent, input_images):
        super(ImportGroundTruth, self).__init__(QIcon('resources/icons/import_image.svg'),
                                                'Import &GroundTruth', parent)
        self.setToolTip("Import GroundTruth files")
        self.triggered.connect(self.perform_action)
        self.input_images = input_images
        self.image_window = parent

    def perform_action(self):
        results = QFileDialog.getOpenFileNames(self.image_window,
                                               "Load Image",
                                               filter="Image Files (*.png)")
        # pylint disable=invalid-name
        for file_name in results[0]:
            try:
                image = imageio.imread(file_name)
                idx = self.parent().item_row
                base_file_name = os.path.basename(file_name)
                self.input_images.add_ground_truth(idx, image, base_file_name)
            # pylint: disable=broad-except
            except Exception:
                QMessageBox.critical(self.image_window,
                                     "The selected file is not supported",
                                     "The file is corrupted or does not contain"
                                     " supported Image format",
                                     QMessageBox.Ok)
