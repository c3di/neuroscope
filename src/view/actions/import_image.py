import os
import imageio
from PySide2.QtWidgets import QAction, QMessageBox, QFileDialog
from PySide2.QtGui import QIcon


class ImportImageAction(QAction):
    def __init__(self, parent, input_images):
        super(ImportImageAction, self).__init__(QIcon('resources/icons/import_image.svg'),
                                                'Import &Images', parent)
        self.setToolTip("Import image files")
        self.triggered.connect(self.perform_action)
        self.input_images = input_images
        self.main_window = parent

    def perform_action(self):
        results = QFileDialog.getOpenFileNames(self.main_window,
                                               "Load Image",
                                               filter="Image Files (*.png *.jpg *.bmp *.gz)")
        # pylint disable=invalid-name
        for file_name in results[0]:
            try:
                image = imageio.imread(file_name)[..., :3]
                base_file_name = os.path.basename(file_name)
                self.input_images.add_image(image, base_file_name)
            # pylint: disable=broad-except
            except Exception:
                QMessageBox.critical(self.main_window,
                                     "The selected file is not supported",
                                     "The file is corrupted or does not contain"
                                     " supported Image format",
                                     QMessageBox.Ok)
        self.main_window.open_image_input_window()
