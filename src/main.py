# -*- coding: utf-8 -*-

import sys
import time
from PySide2 import QtGui, QtCore
from PySide2.QtCore import QFile, QTextStream, QSize
from PySide2.QtGui import QPixmap, QIcon
from PySide2.QtWidgets import QSplashScreen, QMainWindow, QDesktopWidget, \
    QMdiArea, QApplication, QAction
from view import SignalHub
from view.windows import InspectionWindow, ArchitectureWindow, \
    PropertiesWindow, InputImageWindow
from view.actions import ImportImageAction, ImportArchitecture, ImportWeights, \
    ArrangeWindowsHorizontal, ArrangeWindowsVertical, ArrangeWindowsTile,\
    ArrangeWindowsCascade, FitNetworkToWindow, SelectGroundTruthDir, QuickStartGuide
from view.context_examples import KerasClassificationExample, KerasSegmentationExample, \
    KerasSegmentationCognitwinExample
from model.MVCModel import MVCMODEL
from model.input_images import InputImages
from controller.inspections import InspectionRegistry


# pylint: disable=too-many-instance-attributes
class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.signal_hub = SignalHub()
        self.inspection_registry = InspectionRegistry()
        self.inspection_windows = []
        self.input_images = InputImages()
        self.document = MVCMODEL
        self.document.images = self.input_images
        self.document.changed.connect(self.signal_hub.on_model_changed)
        self.init_actions()
        self.architecture_dock = None
        self.properties_dock = None
        self.initialize_style_sheet()
        self.initialize_main_window()
        self.initialize_menu_bar()
        self.app = None
        self.image_input_dock = None

    # pylint: disable=attribute-defined-outside-init
    def initialize_main_window(self):
        self.make_window_full_screen()
        self.set_main_window_title()
        self.main_area = QMdiArea()
        self.main_area.setStyleSheet(self.style_sheet)
        self.setCentralWidget(self.main_area)
        self.tool_bar = None
        self.window_menu = None
        self.file_menu = None

    def initialize_style_sheet(self):
        file = QFile("resources/styles/styles.qss")
        if file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text):
            stream = QTextStream(file)
            style_sheet = stream.readAll()
            file.close()
            self.style_sheet = style_sheet
        self.setStyleSheet(self.style_sheet)

    def make_window_full_screen(self):
        desktop = QDesktopWidget()
        geometry = desktop.availableGeometry()
        self.setGeometry(geometry)
        self.showMaximized()

    def set_main_window_title(self):
        self.setWindowTitle(self.tr("Neuroscope – XAI for classification and semantic segmentation"))

    def initialize_menu_bar(self):
        self.menuBar()
        self.add_file_menu()
        self.add_window_menu()
        self.add_example_menu()
        self.add_help_menu()
        self.init_tool_bar()

    def add_file_menu(self):
        self.file_menu = self.menuBar().addMenu('&File')
        self.file_menu.addAction(self.import_network_action)
        self.file_menu.addAction(self.import_images_action)
        self.file_menu.addAction(self.select_ground_truth_dir)
        self.file_menu.addAction(self.exit_application_action)

    def add_window_menu(self):
        self.window_menu = self.menuBar().addMenu('&Window')
        self.window_menu.addAction(self.open_architecture_window_action)
        self.window_menu.addAction(self.add_inspection_action)
        self.window_menu.addAction(self.open_properties_window_action)
        self.window_menu.addAction(self.open_image_input_window_action)

    def add_example_menu(self):
        self.example_menu = self.menuBar().addMenu('&Example')
        self.example_menu.addAction(self.keras_classification_example)
        self.example_menu.addAction(self.keras_segmentation_example)
        self.example_menu.addAction(self.keras_segmentation_Cognitwin_example)

    def add_help_menu(self):
        self.help_menu = self.menuBar().addMenu('&Help')
        self.help_menu.addAction(self.quick_start_guide)

    def init_tool_bar(self):
        self.tool_bar = self.addToolBar('Open')
        self.tool_bar.addAction(self.import_network_action)
        self.tool_bar.addAction(self.import_images_action)
        self.tool_bar.setStyleSheet("QToolBar::"
                                    "separator { background-color: white;"
                                    "width: 3; height: 3; }")
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.window_tile_vertical)
        self.tool_bar.addAction(self.window_tile_horizontal)
        self.tool_bar.addAction(self.window_tile_grid)
        self.tool_bar.addAction(self.window_tile_cascade)
        self.tool_bar.addAction(self.fit_network_to_window_size)
        self.tool_bar.addSeparator()

    def init_actions(self):
        self.import_network_action = ImportArchitecture(self, self.document)
        self.import_weights_action = ImportWeights(self, self.document)
        self.import_images_action = ImportImageAction(self, self.input_images)
        self.select_ground_truth_dir = SelectGroundTruthDir(self, self.input_images)
        self.keras_classification_example = KerasClassificationExample(self, self.document)
        self.keras_segmentation_example = KerasSegmentationExample(self, self.document)
        self.keras_segmentation_Cognitwin_example = KerasSegmentationCognitwinExample(self, self.document)
        self.exit_application_action = QAction(
            QIcon('resources/icons/exit.svg'), 'E&xit', self)
        self.exit_application_action.triggered.connect(self.close)
        self.add_inspection_action = QAction(
            QtGui.QIcon('resources/icons/NewInspect.svg'),
            '&Inspection Window', self)
        self.add_inspection_action.triggered.connect(self.add_inspection_window)
        self.open_architecture_window_action = QAction(
            QtGui.QIcon('resources/images/ModelDock.svg'),
            'Architecture Window', self)
        self.open_architecture_window_action.triggered.connect(
            self.open_architecture_window)
        self.open_properties_window_action = QAction(
            QtGui.QIcon('resources/icons/layers.svg'),
            'Properties Window', self)
        self.open_properties_window_action.triggered.connect(
            self.open_properties_window)
        self.window_tile_vertical = ArrangeWindowsVertical(self)
        self.window_tile_horizontal = ArrangeWindowsHorizontal(self)
        self.window_tile_grid = ArrangeWindowsTile(self)
        self.window_tile_cascade = ArrangeWindowsCascade(self)
        self.fit_network_to_window_size = FitNetworkToWindow(self)
        self.open_image_input_window_action = QAction(
            QtGui.QIcon('resources/icons/layers.svg'),
            'Input Images Window', self)
        self.open_image_input_window_action.triggered.connect(
            self.open_image_input_window)
        self.quick_start_guide = QuickStartGuide(self)

    def add_inspection_window(self):
        if self.architecture_dock is None:
            self.open_architecture_window()
        sub_window = InspectionWindow(input_images=self.input_images,
                                      available_inspections=self.inspection_registry,
                                      parent=self)
        self.main_area.addSubWindow(sub_window)
        sub_window.show()
        self.inspection_windows.append(sub_window)
        self.input_images.added.connect(sub_window.on_input_image_added)
        self.input_images.removed.connect(sub_window.on_input_image_removed)
        self.signal_hub.layer_selected.connect(sub_window.on_layer_selected)
        sub_window.on_layer_selected(self.architecture_dock.get_selected_node())

    def open_architecture_window(self):
        if self.architecture_dock is None:
            self.architecture_dock = ArchitectureWindow(
                self.document, self, self.style_sheet)
            self.architecture_dock.layer_selected.connect(
                self.signal_hub.on_layer_selected)
            self.signal_hub.model_changed.connect(
                self.architecture_dock.on_model_changed)
            self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                               self.architecture_dock)
        self.architecture_dock.show()

    def open_properties_window(self):
        if self.architecture_dock is None:
            self.open_architecture_window()
        if self.properties_dock is None:
            self.properties_dock = PropertiesWindow(self,
                                                    self.style_sheet,
                                                    "Properties")
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea,
                               self.properties_dock)
            self.signal_hub.layer_selected.connect(
                self.properties_dock.on_layer_selected)
            self.signal_hub.model_changed.connect(
                self.properties_dock.on_layer_selected)
            self.properties_dock.on_layer_selected(
                self.architecture_dock.get_selected_node())
        self.properties_dock.show()

    def open_image_input_window(self):
        if self.architecture_dock is None:
            self.open_architecture_window()
        if self.image_input_dock is None:
            self.image_input_dock = \
                InputImageWindow(
                    self.input_images, parent=self,
                    style_sheet=self.style_sheet)
            self.addDockWidget(
                QtCore.Qt.LeftDockWidgetArea, self.image_input_dock)
        self.image_input_dock.show()
        self.input_images.added.connect(
            self.image_input_dock.on_input_image_added)
        self.input_images.gt_added.connect(
            self.image_input_dock.on_ground_truth_added
        )
        self.input_images.removed.connect(
            self.image_input_dock.on_image_removed
        )


if __name__ == '__main__':
    # pylint: disable = invalid-name
    app = QApplication(sys.argv)
    # pylint: disable = invalid-name
    pixmap = QPixmap("resources/images/splash.png")
    # pylint: disable = invalid-name
    splash = QSplashScreen(pixmap)
    splash.show()
    time.sleep(2)
    # pylint: disable = invalid-name
    main_window = MainWindow()
    main_window.setWindowIcon(QIcon("resources/images/icon.svg"))
    main_window.app = app
    main_window.show()
    splash.finish(main_window)
    app.exec_()
