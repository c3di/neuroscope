from PySide2.QtCore import QPoint, QRect, Slot
from PySide2.QtWidgets import QAction
from PySide2.QtGui import QIcon


class ArrangeWindowsHorizontal(QAction):
    def __init__(self, parent):
        super(ArrangeWindowsHorizontal, self).__init__(QIcon('resources/icons/'
                                                             'arrange_horizontally.svg'),
                                                       'Arrange Windows Horizontally', parent)
        self.setToolTip("Arrange Windows Horizontally")
        self.triggered.connect(self.perform_action)
        self.main_window = parent

    @Slot()
    def perform_action(self):
        main_area = self.main_window.main_area
        window_count = len(main_area.subWindowList())
        if window_count == 0:
            return

        position = QPoint(0, 0)
        for window in main_area.subWindowList():
            rect = QRect(position.x(), position.y(), main_area.width()/window_count,
                         main_area.height())
            window.setGeometry(rect)
            position.setX(position.x() + window.width())
