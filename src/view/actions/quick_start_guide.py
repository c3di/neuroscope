from PySide2.QtCore import Slot
from PySide2.QtWidgets import QAction
import webbrowser


class QuickStartGuide(QAction):
    def __init__(self, parent):
        super(QuickStartGuide, self).__init__('Quick Start Guide', parent)
        self.setToolTip("QuickStartGuide")
        self.triggered.connect(self.perform_action)
        self.main_window = parent

    @Slot()
    def perform_action(self):
        webbrowser.get('windows-default')
        webbrowser.open_new('https://github.com/c3di/neuroscope/blob/release/Neuroscope.Quick.Start.Guide.pdf')
