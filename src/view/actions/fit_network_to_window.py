from PySide2.QtCore import Slot
from PySide2.QtWidgets import QAction
from PySide2.QtGui import QIcon


class FitNetworkToWindow(QAction):
    def __init__(self, parent):
        super(FitNetworkToWindow, self).__init__(QIcon('resources/icons/fit_network_to_window.svg'),
                                                 'Fit Network To Window', parent)
        self.setToolTip("Fit network to the window")
        self.triggered.connect(self.perform_action)
        self.main_window = parent

    @Slot()
    def perform_action(self):
        main_area = self.main_window
        if main_area.architecture_dock is not None:
            main_area.architecture_dock.fit_to_window_size()
