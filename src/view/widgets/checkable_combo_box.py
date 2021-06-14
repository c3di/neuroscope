from PySide2.QtCore import Signal, Qt
from PySide2.QtWidgets import QComboBox


class CheckableComboBox(QComboBox):

    itemSelected = Signal()

    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)
        self.model().dataChanged.connect(self.on_checked)

    def add_item(self, item_to_add):
        super(CheckableComboBox, self).addItem(item_to_add)
        item = self.model().item(self.count()-1, 0)
        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.setCheckState(Qt.Unchecked)

    def on_checked(self):
        self.itemSelected.emit()
