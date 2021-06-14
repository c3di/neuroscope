from PySide2.QtWidgets import QGraphicsSimpleTextItem


class CenteredGraphicsTextItem(QGraphicsSimpleTextItem):
    def __init__(self, text):
        super(CenteredGraphicsTextItem, self).__init__(text)

    # pylint: disable=invalid-name, unused-argument
    def paint(self, painter, option, widget=None):
        rect = self.boundingRect()
        width = rect.right()
        height = rect.bottom()
        painter.translate(-width/2, -height/2)
        super().paint(painter, option, widget)
