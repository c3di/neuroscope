from PySide2.QtWidgets import QStackedWidget
from view.widgets import ImageWidget
from view.widgets import MultiImageWidget


class AnyImageWidget(QStackedWidget):
    def __init__(self, parent=None):
        super(AnyImageWidget, self).__init__(parent)
        self.single_image_view = ImageWidget()
        self.multi_image_view = MultiImageWidget()
        self.init_layout()

    def init_layout(self):
        self.addWidget(self.single_image_view)
        self.addWidget(self.multi_image_view)

    def set_content(self, content):
        if content is None:
            self.hide()
        elif len(content) == 1:
            self.show()
            self.setCurrentIndex(0)
            self.single_image_view.set_image(content[0])
        else:
            self.show()
            self.setCurrentIndex(1)
            self.multi_image_view.set_content(content)

    def current_row(self):
        if isinstance(self.currentWidget(), MultiImageWidget):
            return self.currentWidget().currentRow()
        else:
            return 0
