from PySide2.QtCore import Slot
from PySide2.QtWidgets import QAction
from PySide2.QtGui import QIcon


class ArrangeWindowsTile(QAction):
    def __init__(self, parent):
        super(ArrangeWindowsTile, self).__init__(QIcon('resources/icons/arrange_in_grid.svg'),
                                                 'Arrange Windows (Tile)', parent)
        self.setToolTip("Arrange Windows by Tiling")
        self.triggered.connect(self.perform_action)
        self.main_window = parent

    @Slot()
    def perform_action(self):
        main_area = self.main_window.main_area
        main_area.tileSubWindows()
