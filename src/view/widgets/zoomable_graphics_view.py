from PySide2.QtWidgets import QGraphicsView
from PySide2.QtCore import QSize


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.zoom_factor = 2
        self.accumulate_zoom = 1
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.verticalScrollBar().hide()
        self.horizontalScrollBar().hide()

    # pylint: disable=invalid-name
    def sizeHint(self):
        return QSize(400, 0)

    # pylint: disable=invalid-name, too-many-branches
    def wheelEvent(self, event):
        if event.angleDelta().y() < 0:
            if self.accumulate_zoom > 0.1:
                zoomfactor = 1.0
                if self.accumulate_zoom / self.zoom_factor < 0.1:
                    zoomfactor = 0.1 / self.accumulate_zoom
                    self.accumulate_zoom = 0.1
                else:
                    self.accumulate_zoom = self.accumulate_zoom / self.zoom_factor
                    zoomfactor = 1.0 / self.zoom_factor
                self.scale(zoomfactor, zoomfactor)
        else:
            if self.accumulate_zoom < 10:
                zoomfactor = 1.0
                if self.accumulate_zoom * self.zoom_factor > 10.0:
                    zoomfactor = 10.0 / self.accumulate_zoom
                    self.accumulate_zoom = 10.0
                else:
                    self.accumulate_zoom = self.accumulate_zoom * self.zoom_factor
                    zoomfactor = self.zoom_factor
                self.scale(zoomfactor, zoomfactor)
