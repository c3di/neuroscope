# -*-coding:utf-8-*-
import os
import re
import copy
from datetime import datetime
import imageio
import numpy as np
from PySide2 import QtCore
from PySide2.QtWidgets import QMdiSubWindow, QComboBox, QLineEdit, \
    QGroupBox, QBoxLayout, QLabel, QHBoxLayout, QMessageBox, QPushButton, QFileDialog
from PySide2.QtGui import QIcon
from model.core import Layer
from view.widgets import AnyImageWidget
from controller.inspections import PredictInspection
from controller.inspections import InputInspection
from controller.backends import ImageFilter
from model.MVCModel import MVCMODEL
from view.windows import InspectionSettingsDialog
from controller.inspections import Settings


class InspectionWindow(QMdiSubWindow):

    def __init__(self, input_images=None, available_inspections=None,
                 parent=None, style_sheet=None):
        super(InspectionWindow, self).__init__(parent)
        self.settings = Settings()
        self.selected_layer = None
        self.image = None
        self.label = 0
        self.predictions = []
        self.parent = parent
        self.image_filter_tool = ImageFilter()
        self.predict = PredictInspection()
        self.input_images = input_images
        self.available_inspections = available_inspections
        self.inspection = self.available_inspections.get_by_index(index=0)
        self.style_sheet = style_sheet
        self.init_widgets()
        self.init_input_selection_items()
        self.image_prediction()
        self.on_layer_selected(None)
        self.init_inspection_items()
        self.init_image_filter()
        self.update_image_display()
        self.init_signal_connections()
        self.update_title()
        self.result_collection = None
        self.settings_window = None
        self.prediction_btn.clicked.connect(self.image_prediction)
        self.settings_btn.clicked.connect(self.open_settings_dialog)

    def init_widgets(self):
        self.image_label = QLabel("Image")
        self.input_selection = QComboBox()
        self.input_selection.setObjectName("input_selections")
        self.input_selection.setMinimumWidth(190)
        self.prediction_btn = QPushButton("Prediction")
        self.prediction_btn.setToolTip("update the prediction when you change the model")
        self.prediction_selection = QComboBox()
        self.prediction_selection.setObjectName("prediction_selections")
        self.prediction_selection.setMinimumWidth(250)
        self.inspection_label = QLabel("Inspection")
        self.operator_selection = QComboBox()
        self.operator_selection.setObjectName("operator_selection")
        self.operator_selection.setMinimumWidth(190)
        self.filter_label = QLabel("Filter")
        self.image_filter = QComboBox()
        self.image_filter.setObjectName("image filter")
        self.layer_selection = QLineEdit()
        self.layer_selection.setReadOnly(True)
        self.layer_selection.setObjectName("layer_selection")
        self.image_display = AnyImageWidget(None)
        self.image_display.setObjectName("image_display")
        self.image_display.setObjectName("image_display")
        self.settings_btn = QPushButton("Settings")
        self.save_result_btn = QPushButton("Save")
        self.init_layout()
        self.setStyleSheet(self.style_sheet)
        self.setWindowIcon(QIcon("resources/images/icon.svg"))

    def init_signal_connections(self):
        self.input_selection.currentIndexChanged.connect(self.on_input_image_selected)
        self.prediction_selection.currentIndexChanged.connect(self.on_prediction_selected)
        self.operator_selection.currentIndexChanged.connect(self.on_inspection_operation_selected)
        self.prediction_btn.clicked.connect(self.image_prediction)
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        self.save_result_btn.clicked.connect(self.save_results)

    def signal_disconnections(self):
        self.input_selection.currentIndexChanged.disconnect()
        self.prediction_selection.currentIndexChanged.disconnect()
        self.operator_selection.currentIndexChanged.disconnect()
        self.prediction_btn.clicked.disconnect()
        self.settings_btn.clicked.disconnect()
        self.save_result_btn.clicked.disconnect()

    def init_layout(self):
        central_widget = QGroupBox()
        self.setWidget(central_widget)
        tool_layout = QBoxLayout(QBoxLayout.TopToBottom)
        central_widget.setLayout(tool_layout)
        tool_bar_row1_layout = QHBoxLayout()
        tool_bar_row1_layout.addWidget(self.image_label)
        tool_bar_row1_layout.addWidget(self.input_selection)
        tool_bar_row1_layout.addWidget(self.prediction_btn)
        tool_bar_row1_layout.addWidget(self.prediction_selection)
        tool_bar_row1_layout.addWidget(self.inspection_label)
        tool_bar_row1_layout.addWidget(self.operator_selection)
        tool_layout.addLayout(tool_bar_row1_layout)
        tool_bar_row2_layout = QHBoxLayout()
        tool_bar_row2_layout.addWidget(self.filter_label)
        tool_bar_row2_layout.addWidget(self.image_filter)
        tool_bar_row2_layout.addWidget(self.save_result_btn)
        tool_bar_row2_layout.addWidget(self.settings_btn)
        tool_layout.addLayout(tool_bar_row2_layout)
        tool_layout.addWidget(self.image_display)

    def init_input_selection_items(self):
        if self.input_images:
            self.image = self.input_images.get_image(0)
            for name in self.input_images.all_image_names():
                self.input_selection.addItem(name)
        else:
            self.image = None

    def init_prediction_selection_items(self):
        self.prediction_selection.clear()
        if self.predictions is not None and len(self.predictions) != 0:
            for _p in self.predictions:
                self.prediction_selection.addItem(_p.name)

    def init_inspection_items(self):
        self.inspection = None
        if MVCMODEL.model is None or not self.available_inspections:
            return
        self.inspection = self.available_inspections.get_by_index(0, context=MVCMODEL.model.context)
        for name in self.available_inspections.all_inspection_names(context=MVCMODEL.model.context):
            self.operator_selection.addItem(name)

    def init_image_filter(self):
        if self.inspection is not None:
            self.update_image_filter()

    def update_title(self):
        caption = self.inspection.caption if self.inspection is not None else ''
        title = "[" + caption + "] [" + self.selected_layer_name() + "]"
        self.setWindowTitle(title)

    def update_image_filter(self, run_first=False):
        try:
            self.image_filter.currentIndexChanged.disconnect()
        # pylint: disable=broad-except
        except Exception:
            pass
        self.image_filter.clear()
        if self.inspection.image_filters is not None:
            for idx, _f in enumerate(self.inspection.image_filters):
                self.image_filter.addItem(_f)
                self.image_filter.setItemData(idx, "This is a tooltip for item[0]", QtCore.Qt.ToolTipRole)
        if run_first:
            self.update_image_display()

        self.image_filter.currentIndexChanged.connect(self.on_image_filter_selected)

    def selected_layer_name(self):
        if self.selected_layer is None:
            return ""
        return self.selected_layer.name

    def update_image_display(self):
        if self.image is None or self.inspection is None:
            self.prediction_selection.clear()
            self.image_display.set_content(None)
            return
        if self.predictions is not None:
            selected_prediction = self.predictions[self.label]
        else:
            selected_prediction = None
        layer_index = self.selected_layer.index if self.selected_layer is not None else None
        message, results = self.inspection.perform(self.image, layer_index, selected_prediction, self.settings)
        if results is None:
            QMessageBox.critical(self, 'error', message, QMessageBox.Ok)
            return
        self.result_collection = results
        filters = self.inspection.image_filters
        _f = None
        if filters is not None:
            _f = filters[self.image_filter.currentIndex()]
        self.image_filter_process(_f)

    def image_filter_process(self, image_filter):
        if self.result_collection is None:
            return
        selected_item_index = self.image_display.current_row()
        selected_item_index = 0 if selected_item_index == -1 else selected_item_index
        current_result = copy.deepcopy(self.result_collection)
        if image_filter is not None and current_result[0] is not None:
            current_result = self.image_filter_tool(image_filter, current_result, **{'image': self.image, 'prediction': self.predictions[-1],
                                                                                     'output_classes': MVCMODEL.model.output_shape[0],
                                                                                     'prediction_index': self.label, 'item_index' : selected_item_index})
        self.image_display.set_content(np.uint8(current_result)) # number * image width * image  height (* channels)
        self.filtered_results = np.uint8(current_result)

    def is_active_mdi_child(self):
        return self == self.parent.main_area.currentSubWindow()

    # pylint: disable=invalid-name, unused-argument
    def closeEvent(self, *args, **kwargs):
        main_window = self.mdiArea().parent()
        main_window.inspection_windows.remove(self)
        main_window.main_area.removeSubWindow(self)
        self.signal_disconnections()
        self.deleteLater()

    def init_settings_dialog(self):
        self.settings_window = InspectionSettingsDialog(title="Setting", parent=self)
        self.settings_window.inspection_settings_changed.connect(self.on_inspection_settings_changed)

    def open_settings_dialog(self):
        if self.settings_window is None:
            self.init_settings_dialog()
        self.settings_window.show()
        self.settings_window.raise_()

    @QtCore.Slot()
    def image_prediction(self):
        if MVCMODEL.model is None:
            return
        if self.image is not None:
            _, self.predictions = self.predict.perform(self.image)
        self.init_prediction_selection_items()
        self.label = self.prediction_selection.currentIndex()

    @QtCore.Slot()
    def save_results(self):
        def get_dir_address_from_user():
            return QFileDialog.getExistingDirectory(self, "select directory to save")

        def make_file_address(dir_path, index):
            file_name, file_extension = os.path.splitext(self.image[1])
            file_name = file_name + '_' + self.prediction_selection.currentText() + '_' \
                        + self.windowTitle() + '_' + self.image_filter.currentText() + '_' + str(index) \
                        + str(datetime.now())
            file_name = re.sub('[^a-zA-Z0-9_]', '_', file_name) + ".png"
            return os.path.join(dir_path, file_name)

        dir_address = get_dir_address_from_user()
        if dir_address == '':
            return
        for index, image in enumerate(self.filtered_results):
            file_address = make_file_address(dir_address, index)
            imageio.imwrite(file_address, image, format="png")

        QMessageBox.information(self, 'Done', "The image has been saved.", QMessageBox.Ok)

    # pylint: disable=invalid-name, unused-argument
    def focusInEvent(self, event):
        self.parent.architecture_dock.select_node_by_name(self.selected_layer_name())

    @QtCore.Slot(np.ndarray, str)
    def on_input_image_added(self, name):
        self.input_selection.addItem(name)
        ind = self.input_selection.count() - 1
        self.input_selection.setCurrentIndex(ind)

    @QtCore.Slot()
    def on_input_image_removed(self, index):
        self.input_selection.removeItem(index)

    @QtCore.Slot(int)
    def on_input_image_selected(self, index):
        self.image = self.input_images.get_image(index)
        prediction_index = self.prediction_selection.currentIndex()
        self.image_prediction()
        self.prediction_selection.setCurrentIndex(prediction_index)
        self.update_image_display()
        self.update_title()

    @QtCore.Slot(int)
    def on_prediction_selected(self, index):
        self.label = index
        if not isinstance(self.inspection, InputInspection):
            self.update_image_display()
        self.update_title()

    @QtCore.Slot(int)
    def on_image_filter_selected(self, index):
        filters = self.inspection.image_filters
        image_filter = None
        if filters is not None:
            image_filter = self.inspection.image_filters[index]
        self.image_filter_process(image_filter)
        self.update_title()

    @QtCore.Slot(int)
    def on_inspection_operation_selected(self, index):
        if MVCMODEL.model is None or not self.available_inspections:
            return
        self.inspection = self.available_inspections.get_by_index(index, context=MVCMODEL.model.context)
        self.update_image_filter(run_first=True)
        self.update_title()

    @QtCore.Slot(Layer)
    def on_layer_selected(self, layer=None):
        if layer is None:
            return
        if not self.is_active_mdi_child():
            return
        self.selected_layer = layer
        self.layer_selection.setText(layer.name)
        if not isinstance(self.inspection, InputInspection):
            self.update_image_display()
        self.update_title()

    @QtCore.Slot(int)
    def on_inspection_settings_changed(self, setting):
        self.settings = setting
        self.image_filter_tool.cmap_color = self.settings.color_mapping
        self.update_image_display()
