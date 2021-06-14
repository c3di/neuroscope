from PySide2.QtCore import Qt
from PySide2.QtGui import QFont
from PySide2.QtWidgets import QGraphicsItemGroup
from model.core import Domain_Layer_Types
from view.scenegraph import CenteredGraphicsSvgItem
from view.scenegraph import CenteredGraphicsTextItem


class SceneGraphLayerItem(QGraphicsItemGroup):
    def __init__(self, node, renderer, parent_window):
        super(SceneGraphLayerItem, self).__init__()
        self.node = node
        self.setPos(self.node.position[0], self.node.position[1])
        self.parent_window = parent_window
        self.renderer = renderer
        self.selected = False
        self.cursor_item = None
        self.label_item = None
        self.box_item = None
        self.add_box_item()
        self.add_label_item()
        self.setAcceptHoverEvents(True)

    def add_box_item(self):
        self.box_item = CenteredGraphicsSvgItem(self.node)
        self.box_item.setSharedRenderer(self.renderer)
        self.box_item.setPos(self.node.position[0], self.node.position[1])
        self.box_item.setScale(1.5)
        self.box_item.setZValue(1)
        self.set_type()
        self.box_item.setAcceptedMouseButtons(Qt.NoButton)
        self.addToGroup(self.box_item)

    def add_label_item(self):
        self.label_item = CenteredGraphicsTextItem(self.node.name)
        self.label_item.setPos(self.node.position[0], self.node.position[1])
        self.label_item.setFont(QFont("Helvetica", 6))
        self.label_item.setZValue(2)
        self.label_item.setAcceptedMouseButtons(Qt.NoButton)
        self.addToGroup(self.label_item)

    def set_type(self):
        if hasattr(self.node, "layer_class"):
            self.node.class_name = self.node.layer_class
            if Domain_Layer_Types.has_value(self.node.layer_class):
                self.box_item.setElementId(self.node.layer_class)
            else:
                self.box_item.setElementId("Default")
                self.node.class_name = "Default"

    def select(self):
        self.selected = True
        self.cursor_item = CenteredGraphicsSvgItem(None)
        self.cursor_item.setSharedRenderer(self.renderer)
        self.cursor_item.setPos(self.node.position[0], self.node.position[1])
        self.cursor_item.setScale(1.5)
        self.cursor_item.setZValue(3)
        self.cursor_item.setElementId("Cursor")
        self.cursor_item.setAcceptedMouseButtons(Qt.NoButton)
        self.addToGroup(self.cursor_item)
        self.box_item.setElementId(self.node.class_name)
        self.update(self.boundingRect())

    def unselect(self):
        self.selected = False
        self.removeFromGroup(self.cursor_item)
        self.cursor_item.hide()
        self.cursor_item = None
        self.update(self.boundingRect())

    # pylint: disable=invalid-name, unused-argument
    def hoverEnterEvent(self, event):
        if not self.selected:
            self.box_item.setElementId(self.node.class_name + "_selected")
            self.update(self.boundingRect())

    # pylint: disable=invalid-name, unused-argument
    def hoverLeaveEvent(self, event):
        self.box_item.setElementId(self.node.class_name)
        self.update(self.boundingRect())

    def mousePressEvent(self, event):
        return

    # pylint: disable=invalid-name, unused-argument
    def mouseReleaseEvent(self, event):
        item = self.parent_window.scene.mouseGrabberItem()
        self.parent_window.select_node(item)
