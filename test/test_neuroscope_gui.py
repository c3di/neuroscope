import unittest
import sys
import warnings
import os.path
import imageio
import numpy as np

from PySide2.QtWidgets import QToolBar, QMdiArea, QApplication
from PySide2.QtTest import QTest
from PySide2.QtGui import QTransform, QCursor
from PySide2.QtCore import QPoint, Qt

from test_setting import TestSetting

from main import MainWindow
from view.scenegraph import SceneGraphLayerItem
from view.scenegraph import CenteredGraphicsTextItem
from controller.backends import BACKEND_PROVIDER

current_path = os.getcwd()
project_path = os.path.abspath(os.path.join(current_path, os.pardir))
os.chdir(project_path)


class TestNeuroscopeGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.setting = TestSetting()
        try:
            cls.app = QApplication(sys.argv)
        except Exception as ex:
            print("error: {0}".format(ex))

    def setUp(self):
        warnings.simplefilter('ignore', category=Warning)
        self.form = MainWindow()

    def tearDown(self):
        self.form.close()

    def test_default(self):
        self.assertEqual(hasattr(self.form, 'file_menu'), True)
        self.assertEqual(hasattr(self.form, 'window_menu'), True)
        self.assertEqual(hasattr(self.form, 'import_network_action'), True)
        self.assertEqual(hasattr(self.form, 'import_weights_action'), True)
        self.assertEqual(hasattr(self.form, 'import_images_action'), True)
        self.assertEqual(hasattr(self.form, 'exit_application_action'), True)
        self.assertEqual(hasattr(self.form, 'open_architecture_window_action'), True)
        self.assertEqual(hasattr(self.form, 'add_inspection_action'), True)
        self.assertEqual(hasattr(self.form, 'open_properties_window_action'), True)
        self.assertEqual(hasattr(self.form, 'inspection_windows'), True)
        self.assertIsNotNone(self.form.style_sheet)
        self.assertIsInstance(self.form.centralWidget(), QMdiArea)
        self.assertIsInstance(self.form.tool_bar, QToolBar)
        self.assertEqual(self.form.inspection_registry.available_inspections['Classification'][0].caption, "Input Image")
        self.assertEqual(self.form.inspection_registry.available_inspections['Classification'][6].caption, "Grad Cam Plus")

    def test_open_properties_window(self):
        self.form.open_properties_window()
        self.assertEqual(self.form.properties_dock.windowTitle(), "Properties")
        self.assertEqual(self.form.dockWidgetArea(self.form.properties_dock),
                         Qt.RightDockWidgetArea)

    def test_open_architecture_window(self):
        self.form.open_architecture_window()
        self.assertEqual(self.form.architecture_dock.windowTitle(), "Architecture")

    def test_add_inspection_window(self):
        self.form.add_inspection_window()
        self.assertEqual(self.form.inspection_windows[0] is not None, True)

    def test_import_image(self):
        self.form.add_inspection_window()
        self.load_sample_image()
        self.assertEqual(self.form.inspection_windows[0].input_selection.currentText(),
                         self.form.input_images.get_image_name(0))

    # @unittest.skip("It is using a large model-file; to run, remove this decoration.")
    def test_import_image_and_prediction(self):
        self.load_sample_model_architecture(self.setting.model_address)
        self.form.add_inspection_window()
        self.load_sample_image()
        self.assertEqual(self.form.inspection_windows[0].input_selection.currentText(),
                         self.form.input_images.get_image_name(0))
        self.assertEqual(self.form.inspection_windows[0].prediction_selection.count(),
                         self.setting.prediction_count)
        self.assertEqual(self.form.inspection_windows[0].prediction_selection.itemText(0),
                         self.setting.prediction_vgg16_cat_dog_0)
        self.assertEqual(self.form.inspection_windows[0].prediction_selection.itemText(3),
                         self.setting.prediction_vgg16_cat_dog_3)

    def test_add_two_inspection_window(self):
        self.form.add_inspection_window()
        self.form.add_inspection_window()
        self.assertEqual(self.form.inspection_windows[1] is not None, True)

    def test_init_backend_provider(self):
        backend = BACKEND_PROVIDER.get_current_backend()
        self.assertEqual(type(backend).__name__, "KerasBackend")

    def test_load_architecture(self):
        self.load_sample_model_architecture(self.setting.network_address)
        item = self.select_node_in_architecture_window_by_name(self.setting.input_layer_name)
        pos = item.group().scenePos()
        item_at_pos = self.form.architecture_dock.scene.itemAt(pos, QTransform())
        items_in_scene = self.form.architecture_dock.scene.items()
        self.assertEqual(len(items_in_scene), self.setting.number_of_item)
        self.assertEqual(item.group(), item_at_pos.group())

    def test_load_properties(self):
        self.load_sample_model_architecture(self.setting.network_address)
        item = self.select_node_in_architecture_window_by_name(self.setting.input_layer_name)
        layer = item.group().node
        self.form.open_properties_window()
        self.form.properties_dock.on_layer_selected(layer)
        tree_item = self.form.properties_dock.property_tree.findItems(
            self.setting.properties_root_name, Qt.MatchContains)[0]
        self.assertEqual(tree_item.childCount(), self.setting.properties_length)
        self.assertEqual(tree_item.text(0), self.setting.properties_root_name)
        self.assertEqual(tree_item.child(self.setting.properties_length-1).text(1),
                         self.setting.input_layer_name)

    def test_update_inspection_window(self):
        self.load_sample_model_architecture(self.setting.network_address)
        item = self.select_node_in_architecture_window_by_name(self.setting.input_layer_name)
        self.form.add_inspection_window()
        self.form.main_area.setActiveSubWindow(self.form.inspection_windows[0])
        layer = item.group().node
        self.form.main_area.activeSubWindow().on_layer_selected(layer)
        self.assertEqual(self.form.inspection_windows[0].windowTitle(),
                         '[Input Image] [' + self.setting.input_layer_name + ']')

    # pylint: disable=invalid-name
    def test_update_inspection_window_all_nodes(self):
        self.load_sample_model_architecture(self.setting.network_address)
        items_in_scene = self.form.architecture_dock.scene.items()
        self.form.add_inspection_window()
        self.form.inspection_windows[0].mdiArea().\
            setActiveSubWindow(self.form.inspection_windows[0])
        for item in items_in_scene:
            if isinstance(item, CenteredGraphicsTextItem):
                layer = item.group().node
                self.form.inspection_windows[0].on_layer_selected(layer)
                inspection_window = self.form.main_area.activeSubWindow()
                title = "[" + inspection_window.inspection.caption + "] [" +\
                        inspection_window.selected_layer_name() + "]"
                self.assertEqual(inspection_window.windowTitle(), title)

    # pylint: disable=invalid-name
    def test_update_inspection_window_all_nodes_using_mouse(self):
        self.load_sample_model_architecture(self.setting.network_address)
        self.form.architecture_dock.view.scale(1, 1)
        self.form.add_inspection_window()
        self.form.inspection_windows[0].mdiArea().\
            setActiveSubWindow(self.form.inspection_windows[0])
        inspection_window = self.form.main_area.activeSubWindow()
        items_in_scene = self.form.architecture_dock.scene.items()
        QTest.mouseMove(self.form, QPoint(0, 0), 0)
        QTest.mouseMove(self.form.architecture_dock, QPoint(0, 0), 0)
        view_pos = QCursor.pos()

        for item in items_in_scene:
            if isinstance(item, CenteredGraphicsTextItem):
                pos = item.group().scenePos()
                mapped_pos = self.form.architecture_dock.view.mapFromScene(pos)
                QTest.mouseMove(self.form.childAt(view_pos), mapped_pos, 0)
                QTest.mousePress(self.form.childAt(view_pos), Qt.LeftButton,
                                 Qt.NoModifier, mapped_pos, 0)
                QTest.mouseRelease(self.form.childAt(view_pos), Qt.LeftButton,
                                   Qt.NoModifier, mapped_pos, 0)
                title = "[" + inspection_window.inspection.caption + "] [" +\
                        inspection_window.selected_layer_name() + "]"
                self.assertEqual(inspection_window.windowTitle(), title)

    # pylint: disable=invalid-name
    def test_load_properties_using_mouse(self):
        self.load_sample_model_architecture(self.setting.network_address)
        self.form.architecture_dock.view.scale(1, 1)
        self.form.open_properties_window()
        items_in_scene = self.form.architecture_dock.scene.items()
        QTest.mouseMove(self.form, QPoint(0, 0), 0)
        QTest.mouseMove(self.form.architecture_dock, QPoint(0, 0), 0)
        view_pos = QCursor.pos()
        tree_item = self.form.properties_dock.property_tree
        for item in items_in_scene:
            if isinstance(item, CenteredGraphicsTextItem):
                pos = item.group().scenePos()
                mapped_pos = self.form.architecture_dock.view.mapFromScene(pos)
                QTest.mouseMove(self.form.childAt(view_pos), mapped_pos, 0)
                QTest.mousePress(self.form.childAt(view_pos), Qt.LeftButton,
                                 Qt.NoModifier, mapped_pos, 0)
                QTest.mouseRelease(self.form.childAt(view_pos), Qt.LeftButton,
                                   Qt.NoModifier, mapped_pos, 0)
                self.assertEqual(tree_item.topLevelItem(0).childCount(),
                                 len(item.group().node.config))
                self.assertEqual(tree_item.topLevelItem(0).text(0),
                                 item.group().node.layer_class)
                if tree_item.topLevelItem(0).child(0).text(0) == 'Name':
                    self.assertEqual(tree_item.topLevelItem(0).child(0).text(1),
                                     item.group().node.name)

    def load_sample_model_architecture(self, address=None):
        BACKEND_PROVIDER.activate_backend("Keras")
        backend = BACKEND_PROVIDER.get_current_backend()
        _, extension = os.path.splitext(address)
        file_type = 'model' if extension == ".h5" else 'architecture'
        model = backend.load(address, file_type)
        self.form.document.model = model
        self.form.open_architecture_window()

    def load_sample_image(self):
        image = imageio.imread(self.setting.image_name)
        image = np.asarray(image)[:, :, :3]
        baseFileName = os.path.basename(self.setting.image_name)
        if self.form.document.model is not None:
            image_shape = np.asarray(image.shape)
            # channel first
            self.form.document.model.input_shape = np.array([image_shape[2], image_shape[0], image_shape[1]])
            self.form.document.model.context = "Classification"
            self.form.document.model.image_is_rgb = False
            self.form.document.model.mapping_file_path = self.setting.mapping_file_path
        self.form.input_images.add_image(image, baseFileName)

    def select_node_in_architecture_window_by_name(self, node_name=None):
        items_in_scene = self.form.architecture_dock.scene.items()
        item = None
        for item in items_in_scene:
            if isinstance(item, CenteredGraphicsTextItem):
                if item.text() == node_name:
                    item.group().scenePos()
                    break
        return item

    # pylint: disable=invalid-name
    def test_iterate_nodes_in_architecture_window(self):
        self.load_sample_model_architecture(self.setting.network_address)
        items_in_scene = self.form.architecture_dock.scene.items()
        for item in items_in_scene:
            if isinstance(item, SceneGraphLayerItem):
                pos = item.scenePos()
            else:
                if hasattr(item, 'scenePos') and item.group():
                    pos = item.group().scenePos()
                    item = item.group()
                else:
                    continue
            item_at_pos = self.form.architecture_dock.scene.itemAt(pos, QTransform())
            self.assertEqual(item, item_at_pos.group())

    # pylint: disable=invalid-name
    def test_iterate_all_text_in_architecture_window(self):
        self.load_sample_model_architecture(self.setting.network_address)
        items_in_scene = self.form.architecture_dock.scene.items()
        for item in items_in_scene:
            if isinstance(item, CenteredGraphicsTextItem):
                pos = item.group().scenePos()
                item_at_pos = self.form.architecture_dock.scene.itemAt(pos, QTransform())
                self.assertEqual(item.group(), item_at_pos.group())

    # pylint: disable=invalid-name
    def test_iterate_all_text_in_architecture_window_using_mouse(self):
        self.load_sample_model_architecture(self.setting.network_address)
        self.form.architecture_dock.view.scale(1, 1)
        items_in_scene = self.form.architecture_dock.scene.items()
        QTest.mouseMove(self.form, QPoint(0, 0), 0)
        QTest.mouseMove(self.form.architecture_dock, QPoint(0, 0), 0)
        view_pos = QCursor.pos()
        for item in items_in_scene:
            if isinstance(item, CenteredGraphicsTextItem):
                node_name = item.text()
                pos = item.group().scenePos()
                mapped_pos = self.form.architecture_dock.view.mapFromScene(pos)
                QTest.mouseMove(self.form.childAt(view_pos), mapped_pos, 0)
                QTest.mousePress(self.form.childAt(view_pos), Qt.LeftButton,
                                 Qt.NoModifier, mapped_pos, 0)
                QTest.mouseRelease(self.form.childAt(view_pos), Qt.LeftButton,
                                   Qt.NoModifier, mapped_pos, 0)
                if (item.group().pos()) != self.form.architecture_dock.selected_node.pos():
                    print(node_name + str(self.form.architecture_dock.selected_node))
                self.assertEqual(item.group(), self.form.architecture_dock.selected_node)

    def sample_mouse_click(self):
        self.load_sample_model_architecture()
        QTest.mouseMove(self.form, QPoint(0, 0), 1000)
        QTest.mousePress(self.form.childAt(100, 100), Qt.LeftButton, Qt.NoModifier,
                         QPoint(216, 450), 1000)
        QTest.mouseRelease(self.form.childAt(100, 100), Qt.LeftButton, Qt.NoModifier,
                           QPoint(216, 450), 1000)
        input("Press Enter to continue...")


if __name__ == '__main__':
    unittest.main()
