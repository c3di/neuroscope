import numpy as np
from PySide2.QtWidgets import QDockWidget, QGroupBox, QMenu, QTableWidget, QTableWidgetItem, QBoxLayout
from PySide2 import QtCore
from PySide2.QtCore import Qt
from view.actions import ImportGroundTruth
from view.actions import RemoveImage


class InputImageWindow(QDockWidget):

    def __init__(self, input_images, parent=None, style_sheet=None):
        super(InputImageWindow, self).__init__(parent)
        self.input_images = input_images
        self.setWindowTitle("Input Images")
        self.setStyleSheet(style_sheet)
        self.central_widget = QGroupBox()
        self.setWidget(self.central_widget)
        self.layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.central_widget.setLayout(self.layout)
        self.image_table = QTableWidget()
        self.image_table.setColumnCount(2)
        self.layout.addWidget(self.image_table)
        self.init_contents()
        self.main_window = parent
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.import_ground_truth = ImportGroundTruth(self, self.input_images)
        self.remove_image = RemoveImage(self, self.input_images)
        self.item = None

    def init_contents(self):
        self.image_table.clear()
        self.image_table.setRowCount(len(self.input_images.images))
        for i, img in enumerate(self.input_images.images):
            self.image_table.setItem(i, 0, QTableWidgetItem(img[1]))
        self.image_table.move(0, 0)
        self.image_table.setHorizontalHeaderLabels(["Image Name", "Ground Truth"])
        self.image_table.show()

    def show_context_menu(self, pos):
        menu = QMenu()
        menu.addAction(self.remove_image)
        menu.addAction(self.import_ground_truth)
        self.item_row = self.image_table.currentRow()
        menu.exec_(self.mapToGlobal(pos))

    def handle_double_click(self, item):
        selected_image_inspection = self.main_window.add_inspection_window()
        for inputs in range(0,
                            selected_image_inspection.input_selection.count()):
            if selected_image_inspection.input_selection.itemText(inputs) == item.text():
                selected_image_inspection.input_selection.setCurrentIndex(inputs)
                break

    @QtCore.Slot(str, int)
    def on_input_image_added(self, image_name, index):
        row_count = self.image_table.rowCount()
        if row_count == index:
            return
        self.image_table.setRowCount(row_count+1)
        self.image_table.setItem(row_count, 0, QTableWidgetItem(image_name))

    @QtCore.Slot(int, str)
    def on_ground_truth_added(self, ind, gt_name):
        self.image_table.setItem(ind, 1, QTableWidgetItem(gt_name))

    @QtCore.Slot(int)
    def on_image_removed(self, index):
        if index >= self.image_table.rowCount():
            return
        self.image_table.removeRow(index)
