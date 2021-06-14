from PySide2.QtSvg import QGraphicsSvgItem


class CenteredGraphicsSvgItem(QGraphicsSvgItem):
    def __init__(self, node):
        super(CenteredGraphicsSvgItem, self).__init__()
        self.node = node

    # pylint: disable=invalid-name
    def boundingRect(self):
        rect = super().boundingRect()
        rect.moveLeft(-rect.right()/2)
        rect.moveTop(-rect.bottom()/2)
        return rect

    # pylint: disable=invalid-name, unused-argument
    def paint(self, painter, option, widget=None):
        rect = self.boundingRect()
        painter.translate(rect.left(), rect.top())
        super().paint(painter, option, widget)
