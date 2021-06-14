from PySide2.QtWidgets import QAction
from PySide2.QtGui import QIcon


class RemoveImage(QAction):
    def __init__(self, parent, input_images):
        super(RemoveImage, self).__init__(QIcon('resources/icons/Remove_Image.svg'),
                                          '&Remove Image', parent)
        self.setToolTip("Remove imported image")
        self.triggered.connect(self.perform_action)
        self.input_images = input_images
        self.image_window = parent

    def perform_action(self):
        image_index = self.parent().item_row
        self.input_images.remove_image(image_index)
