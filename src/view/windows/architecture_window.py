import subprocess

from PySide2 import QtCore, QtGui
from PySide2.QtGui import QPainterPath
from PySide2.QtCore import Qt, Signal
from PySide2.QtWidgets import QDockWidget, QToolBar, QGroupBox, QBoxLayout,\
    QAction, QGraphicsScene, QGraphicsPathItem, QLabel
from PySide2.QtSvg import QSvgRenderer
from networkx import Graph

from view.scenegraph import SceneGraphLayerItem, CenteredGraphicsTextItem
from view.scenegraph.nx_pydot import pydot_layout
from view.widgets import ZoomableGraphicsView
from view.windows import ArchitectureSettingsDialog
from model.core import Layer
from model.MVCModel import MVCMODEL


class ArchitectureWindow(QDockWidget):

    layer_selected = Signal(Layer)

    def __init__(self, document, parent=None, style_sheet=None):
        super(ArchitectureWindow, self).__init__(parent)
        self.document = document
        self.selected_node = None
        self.setWindowTitle("Architecture")
        self.central_widget = QGroupBox()
        self.setWidget(self.central_widget)
        self.architecture_dialog_button = QAction(QtGui.QIcon('resources/icons/'
                                                              'architecture-options.svg'),
                                                  'Model options...', self)
        self.architecture_dialog_button.triggered.connect(self.open_architecture_dialog)
        self.tool_bar = QToolBar()
        self.tool_bar.addAction(self.architecture_dialog_button)
        self.model_name_label = QLabel()
        self.tool_bar.addWidget(self.model_name_label)
        self.tool_layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.central_widget.setLayout(self.tool_layout)
        self.tool_layout.addWidget(self.tool_bar)
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing|QtGui.QPainter.TextAntialiasing)
        self.sub_window = None
        self.tool_layout.addWidget(self.view)
        self.init_renderer()
        if self.document.model is not None:
            self.scene_populate()
        self.setStyleSheet(style_sheet)
        self.node_position = None
        self.edge_position = None
        self.graph = None

    def init_renderer(self):
        self.renderer = QSvgRenderer("resources/images/architecture_style.svg")

    def scene_populate(self):
        self.create_graph_for_layout()
        self.layout_scene()
        self.add_nodes_to_scene()
        self.add_edges_to_scene()
        self.node_position = None

    def create_graph_for_layout(self):
        self.graph = Graph()
        for layer in self.document.model.layers:
            self.graph.add_node(layer)
        for layer in self.document.model.layers:
            for inbound_layer in layer.inbound_layers:
                self.graph.add_edge(layer, inbound_layer)

    def layout_scene(self):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        self.node_position, self.edge_position = pydot_layout(self.graph, prog='dot')

    def add_nodes_to_scene(self):
        for node in self.graph.nodes():
            node.position = self.node_position[node]
            layer_item = SceneGraphLayerItem(node, self.renderer, self)
            layer_item.setZValue(1.0)
            self.scene.addItem(layer_item)

    def add_edges_to_scene(self):
        for (from_item, to_item) in self.graph.edges():
            edge_item = QGraphicsPathItem()
            path = QPainterPath()
            #path.moveTo(from_item.position[0], from_item.position[1])
            points = self.edge_position[(from_item, to_item)]
            path.moveTo(from_item.position[0], from_item.position[1])
            path.lineTo(points[0][0], points[0][1])
            for i in range(0, len(points)-1):
                if (i%3) == 0:
                    path.moveTo(points[i][0], points[i][1])
                    path.cubicTo(points[i+1][0], points[i+1][1], points[i+2][0],
                                 points[i+2][1], points[i+3][0], points[i+3][1])
            path.moveTo(points[-1][0], points[-1][1])
            path.lineTo(to_item.position[0], to_item.position[1])
            edge_item.setPath(path)
            edge_item.setZValue(0.0)
            self.scene.addItem(edge_item)

    def select_node(self, node):
        if self.selected_node is not None:
            if self.selected_node.selected is True:
                self.selected_node.unselect()
        node.select()
        self.selected_node = node
        self.layer_selected.emit(node.node)

    def select_node_silent(self, node):
        if self.selected_node is not None:
            if self.selected_node.selected is True:
                self.selected_node.unselect()
        node.select()
        self.selected_node = node
        if self.parent().properties_dock is not None:
            self.parent().properties_dock.on_layer_selected(self.get_selected_node())

    def get_selected_node(self):
        if self.selected_node is None:
            return None
        return self.selected_node.node

    def clear_scene(self):
        self.scene.clear()
        self.selected_node = None

    def init_option_dialog(self, title="Model options"):
        self.sub_window = ArchitectureSettingsDialog(title, parent=self)
        self.sub_window.setting_dialog_closed.connect(self.on_option_dialog_close)

    def open_architecture_dialog(self):
        if self.sub_window is None:
            self.init_option_dialog()
        self.sub_window.show()
        self.sub_window.raise_()

    def select_node_by_name(self, node_name=""):
        items_in_scene = self.scene.items()
        for item in items_in_scene:
            if isinstance(item, CenteredGraphicsTextItem):
                if item.text() == node_name:
                    self.select_node_silent(item.group())
                    break

    @QtCore.Slot()
    def on_option_dialog_close(self):
        self.sub_window = None

    @QtCore.Slot()
    def on_model_changed(self):
        self.model_name_label.setText(MVCMODEL.model_name)
        self.clear_scene()
        self.scene_populate()
        rect = self.scene.itemsBoundingRect()
        self.scene.setSceneRect(rect)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.view.accumulate_zoom = 0.1

    @QtCore.Slot()
    def fit_to_window_size(self):
        rect = self.scene.itemsBoundingRect()
        self.scene.setSceneRect(rect)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.view.accumulate_zoom = 0.1
