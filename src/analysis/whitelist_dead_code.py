# pylint: skip-file
# -*-coding:utf-8-*-
# pylint: skip-file
"""
Due to Python's dynamic nature, static code analyzers like
Vulture are likely to miss some dead code.
Also, code that is only called implicitly may be reported as unused.
The way we deal with false positives is
to create a whitelist Python file. In such a whitelist we simulate
the usage of variables, attributes, etc.
"""

from view.windows import InspectionWindow
from view.windows.architecture_settings_dialog import ArchitectureSettingsDialog
from view.widgets.zoomable_graphics_view import ZoomableGraphicsView
from view.widgets.image_widget import ImageWidget
from view.scenegraph.scene_graph_layer_item import SceneGraphLayerItem
from model.core.network import Network
from controller.backends.keras.keras_backend import _register_guided_gradient,\
    _register_rectified_gradient

# gui events
InspectionWindow.closeEvent()
InspectionWindow.focusOutEvent()
InspectionWindow.focusInEvent()
ArchitectureSettingsDialog.closeEvent()
ArchitectureSettingsDialog.wheelEvent()
ZoomableGraphicsView.wheelEvent()
ZoomableGraphicsView.sizeHint()
SceneGraphLayerItem.hoverEnterEvent()
SceneGraphLayerItem.hoverLeaveEvent()
SceneGraphLayerItem.mousePressEvent()
SceneGraphLayerItem.mouseReleaseEvent()
ImageWidget.paintEvent()

Network.input_layer()
_register_guided_gradient._guided_backpro()
_register_rectified_gradient._rectified_backpro()
