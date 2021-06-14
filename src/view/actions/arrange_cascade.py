from PySide2.QtCore import Slot
from PySide2.QtWidgets import QAction
from PySide2.QtGui import QIcon


class ArrangeWindowsCascade(QAction):
    def __init__(self, parent):
        super(ArrangeWindowsCascade, self).__init__(QIcon('resources/icons/arrange_in_cascade.svg'),
                                                    'Arrange Windows (Cascade)', parent)
        self.setToolTip("Arrange Windows Cascade")
        self.triggered.connect(self.perform_action)
        self.main_window = parent

    @Slot()
    def perform_action(self):
        main_area = self.main_window.main_area
        main_area.cascadeSubWindows()
