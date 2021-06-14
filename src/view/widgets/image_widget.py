from PySide2.QtWidgets import QWidget
from PySide2.QtGui import QPainter, QPixmap
import numpy as np
import qimage2ndarray


class ImageWidget(QWidget):
    def __init__(self, image=None, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.pixmap = QPixmap()
        self.set_image(image)
        self.painter = QPainter()

    def image(self):
        return self.img

    def set_float_image(self, image):
        image_min = image.min()
        image_max = image.max()
        image_range = image_max-image_min
        transformed_image = image-image_min
        transformed_image = transformed_image * image_range
        transformed_image = transformed_image * 255
        qimage2ndarray.array2qimage(transformed_image.astype(int))

    def set_image(self, image):
        if image is None:
            return
        elif isinstance(image, tuple):
            image = image[0]

        if image.dtype == np.uint8:
            q_image = qimage2ndarray.array2qimage(image)
        elif image.dtype == np.float32:
            self.set_float_image(image)
            q_image = image
        self.pixmap.convertFromImage(q_image)
        self.repaint()

    # pylint: disable=invalid-name, unused-argument
    def paintEvent(self, event):
        self.painter.begin(self)
        self.painter.setRenderHint(QPainter.Antialiasing)
        img_x = 0.0
        img_y = 0.0
        height_ratio = self.height() / self.pixmap.height() if self.pixmap.height() != 0 else 1
        width_ratio = self.width() / self.pixmap.width() if self.pixmap.width() != 0 else 1
        new_height, new_width = 0, 0
        if height_ratio > width_ratio:
            new_width = self.width()
            new_height = self.pixmap.height() * width_ratio
            img_y = (self.height() - new_height) / 2.0
        else:
            new_height = self.height()
            new_width = self.pixmap.width() * height_ratio
            img_x = (self.width() - new_width) / 2.0
        self.painter.drawPixmap(img_x, img_y, new_width, new_height, self.pixmap)
        self.painter.end()
