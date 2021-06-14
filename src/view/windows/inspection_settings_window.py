from PySide2 import QtCore
from PySide2.QtWidgets import QGridLayout, QPushButton, \
    QDialog, QLabel, QComboBox, QTableWidget, QTableWidgetItem
from PySide2.QtCore import Signal
from PySide2.QtGui import QIcon
from controller.backends import ImageFilter
from controller.inspections import Settings


class InspectionSettingsDialog(QDialog):

    inspection_settings_changed = Signal(Settings)

    def __init__(self, title, parent):
        super(InspectionSettingsDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.main_window = parent
        self.init_layout()
        self.settings = Settings()
        self.setWindowIcon(QIcon("resources/images/icon.svg"))

    def init_layout(self):
        row = 0
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('Color Mapping'), row, 0)
        self.cmap_comboBox = QComboBox()
        self.cmap_comboBox.addItems(ImageFilter.cmap_color_list)
        grid_layout.addWidget(self.cmap_comboBox, row, 1)
        row += 1

        grid_layout.addWidget(QLabel('Selecting classes are effective to some of inception methods, \n Fusion Maps, '
                                     'Similarity Map'), row, 0)
        row += 1
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(1)
        self.class_table.verticalHeader().setVisible(False)
        grid_layout.addWidget(self.class_table, row, 0)
        self.setLayout(grid_layout)
        self.class_table.setHorizontalHeaderLabels(["Classes"])
        self.class_table.setRowCount(len(self.parent().predictions) - 1)
        for i, prediction in enumerate(self.parent().predictions[0:-1]):
            item = QTableWidgetItem(prediction.name)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            item.setCheckState(QtCore.Qt.Checked)
            self.class_table.setItem(i, 0, item)
        self.class_table.horizontalHeader().setStretchLastSection(True)

        row += 1
        self.apply_btn = QPushButton('Apply', self)
        self.apply_btn.clicked.connect(self.apply_values)
        grid_layout.addWidget(self.apply_btn, row, 0)
        close_btn = QPushButton('Cancel', self)
        close_btn.clicked.connect(self.cancel)
        grid_layout.addWidget(close_btn, row, 1)

    def apply_values(self):
        self.settings.color_mapping = self.cmap_comboBox.currentText()

        self.close()
        class_list = [True] * self.class_table.rowCount()

        for i in range(self.class_table.rowCount()):
            class_list[i] = self.class_table.item(i, 0).checkState() == QtCore.Qt.Checked

        self.settings.selected_classes = class_list
        self.inspection_settings_changed.emit(self.settings)

    def cancel(self):
        self.close()
