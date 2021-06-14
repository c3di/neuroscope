import numpy as np
from PySide2.QtCore import QSize, Qt
from PySide2.QtGui import QPixmap, QIcon
from PySide2.QtWidgets import QListWidgetItem, QListWidget
from view.image_adapter import numpy_to_qimage


class MultiImageWidget(QListWidget):
    def __init__(self, images=None, parent=None):
        super(MultiImageWidget, self).__init__(parent)
        self.content = images
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(256, 256))
        self.setUniformItemSizes(True)
        self.setResizeMode(QListWidget.Adjust)

    def set_content(self, content):
        self.content = content
        self.update_content()

    def update_content(self):
        self.clear()
        for idx, image in enumerate(self.content):
            if np.ndim(image) == 0:
                image = np.ones((2, 2)).astype(np.uint8) * image
            qimage = numpy_to_qimage(image)
            if image.shape[0] < 32 or image.shape[1] < 32:
                qimage = qimage.scaled(QSize(32, 32), Qt.KeepAspectRatioByExpanding)
            list_item = QListWidgetItem(QIcon(QPixmap(qimage)), str(idx))
            self.addItem(list_item)
        self.repaint()
