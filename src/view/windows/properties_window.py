from PySide2 import QtCore
from PySide2.QtWidgets import QDockWidget, QTableWidget,\
    QTreeWidget, QTreeWidgetItem
from model.core import Layer


class PropertiesWindow(QDockWidget):
    def __init__(self, parent=None, style_sheet=None, title=""):
        super(PropertiesWindow, self).__init__(parent)
        self.setWindowTitle(title)
        self.property_sheet = QTableWidget()
        self.property_tree = QTreeWidget()
        self.property_sheet.windowFlags()
        self.setWidget(self.property_tree)
        header = QTreeWidgetItem(["Property", "Value"])
        self.property_tree.setHeaderItem(header)
        self.property_sheet.horizontalHeader().hide()
        self.property_sheet.verticalHeader().hide()
        self.property_sheet.setColumnCount(2)
        self.property_sheet.horizontalScrollBar().hide()
        self.property_sheet.setEnabled(False)
        self.setStyleSheet(style_sheet)

    @QtCore.Slot(Layer)
    def on_layer_selected(self, layer=None):
        if layer is None:
            self.property_tree.clear()
            return
        self.property_tree.clear()
        root = QTreeWidgetItem(self.property_tree, [str(layer.layer_class)])
        root.setExpanded(True)
        keys = layer.config.keys()
        self.add_item(root, layer, keys)
        self.property_tree.resizeColumnToContents(0)
        return

    def add_item(self, parent, layer, keys):
        for item_name in keys:
            item_data = layer.config[item_name]
            if isinstance(item_data, dict):
                key_item = QTreeWidgetItem(parent, [item_name.capitalize()])
                self.add_dict_item(key_item, item_data)
            else:
                value_item = str(item_data)
                QTreeWidgetItem(parent, [item_name.capitalize(), value_item])

    def add_dict_item(self, parent, item_data):
        sub_keys = item_data.keys()
        for item_name in sub_keys:
            sub_item_data = item_data[item_name]
            if isinstance(sub_item_data, dict):
                key_item = QTreeWidgetItem(parent, [item_name.capitalize()])
                self.add_dict_item(key_item, sub_item_data)
            else:
                value_item = str(sub_item_data)
                QTreeWidgetItem(parent, [item_name.capitalize(), value_item])
