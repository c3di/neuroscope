from PySide2 import QtCore
from PySide2.QtWidgets import QGridLayout, QPushButton, QLineEdit,\
    QDialog, QLabel, QInputDialog, QComboBox, QHBoxLayout, \
    QCheckBox, QWidget, QLayoutItem
from PySide2.QtCore import Signal
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QFileDialog, QMessageBox
from model.core.model import CONTEXTS_PRESETS, is_segmentation
from model.MVCModel import MVCMODEL
from controller.preprocessing import MODEL_PREPROCESS


class ArchitectureSettingsDialog(QDialog):

    setting_dialog_closed = Signal()

    def __init__(self, title, parent):
        super(ArchitectureSettingsDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.main_window = parent
        self.setGeometry(300, 300, 455, 400)
        self.init_widgets()
        self.init_value()
        self.setWindowIcon(QIcon("resources/images/icon.svg"))

    def init_widgets(self):
        grid_layout = QGridLayout()
        widgets_and_layout = list()
        widgets_and_layout.extend([self.init_contexts(),
                                   self.init_image_dimensions(),
                                   self.init_preprocessing(),
                                   self.init_mean(),
                                   self.init_std(),
                                   self.init_processing_new_button(),
                                   self.init_input_image_normalization(),
                                   self.init_input_image_channelfirst(),
                                   self.init_input_image_isrgb(),
                                   self.init_decoding_file(),
                                   self.init_output_dimensions_when_segmentation(),
                                   self.init_output_is_channelfirst(),
                                   self.init_add_softmax_activation(),
                                   self.init_apply_cancel_button()])
        i = 0
        for obj_in_raw in widgets_and_layout:
            j = 0
            for obj in obj_in_raw:
                if isinstance(obj, QWidget):
                    grid_layout.addWidget(obj, i, j)
                if isinstance(obj, QLayoutItem):
                    grid_layout.addLayout(obj, i, j)
                j += 1
            i += 1
        self.setLayout(grid_layout)

    def init_contexts(self):
        context_label = QLabel('Contexts')
        context_label.setToolTip('Which kind of computer vision task the model are used for')
        self.context_comboBox = QComboBox()
        self.context_comboBox.addItems(CONTEXTS_PRESETS)
        self.context_comboBox.currentIndexChanged.connect(self.context_selected)
        return context_label, self.context_comboBox

    def init_image_dimensions(self):
        input_shape = [3, 0, 0]
        img_label = QLabel('Image dimensions')
        img_label.setToolTip('The channel, width and height of input image when training the model')
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(QLabel('Channel'))
        self.image_channel = QLineEdit(str(input_shape[0]))
        horizontal_layout.addWidget(self.image_channel)
        horizontal_layout.addWidget(QLabel('Width'))
        self.image_width = QLineEdit(str(input_shape[2]))
        horizontal_layout.addWidget(self.image_width)
        horizontal_layout.addWidget(QLabel('Height'))
        self.image_height = QLineEdit(str(input_shape[1]))
        horizontal_layout.addWidget(self.image_height)
        return img_label, horizontal_layout

    def init_preprocessing(self):
        preprocessing_preset_label = QLabel('Preprocessing presets')
        preprocessing_preset_label.setToolTip('The mean and standard Deviation of the dataset are used'
                                              ' when do mean removal or standardization.')
        preprocessing_preset = QComboBox()
        preprocessing_preset.addItems(MODEL_PREPROCESS.get_all_preset_name())
        preprocessing_preset.currentIndexChanged.connect(self.preprocessing_presets_selected)
        return preprocessing_preset_label, preprocessing_preset

    def init_mean(self):
        self.channel_0_in_mean, self.channel_1_in_mean, self.channel_2_in_mean = QLineEdit(), QLineEdit(), QLineEdit()
        horizontal_layout_for_mean = QHBoxLayout()
        horizontal_layout_for_mean.addWidget(self.channel_0_in_mean)
        horizontal_layout_for_mean.addWidget(self.channel_1_in_mean)
        horizontal_layout_for_mean.addWidget(self.channel_2_in_mean)
        return QLabel('Mean'), horizontal_layout_for_mean

    def init_std(self):
        self.channel_0_in_std, self.channel_1_in_std, self.channel_2_in_std = QLineEdit(), QLineEdit(), QLineEdit()
        horizontal_layout_for_std = QHBoxLayout()
        horizontal_layout_for_std.addWidget(self.channel_0_in_std)
        horizontal_layout_for_std.addWidget(self.channel_1_in_std)
        horizontal_layout_for_std.addWidget(self.channel_2_in_std)
        return QLabel('Standard deviation'), horizontal_layout_for_std

    def init_processing_new_button(self):
        new_preprocessing_preset = QPushButton('new')
        new_preprocessing_preset.clicked.connect(self.add_new_preprocessing_presets)
        return None, new_preprocessing_preset

    def init_input_image_normalization(self):
        normolization_label = QLabel('Input image normalization')
        normolization_label.setToolTip('Should image be normalized to [0, 1).')
        self.should_normalization_checkbox = QCheckBox('')
        return normolization_label, self.should_normalization_checkbox

    def init_input_image_channelfirst(self):
        channel_first_label = QLabel('Input image is channel first')
        channel_first_label.setToolTip('channel dimension for the image is at last')
        self.channel_first_checkbox = QCheckBox('')
        return channel_first_label, self.channel_first_checkbox

    def init_input_image_isrgb(self):
        rgb_label = QLabel('Input image is RGB')
        rgb_label.setToolTip('Whether the input image is rgb or grayscale?')
        self.image_is_rgb_checkbox = QCheckBox()
        self.image_is_rgb_checkbox.setChecked(True)
        return rgb_label, self.image_is_rgb_checkbox

    def init_decoding_file(self):
        decodeing_file_label = QLabel('Decoding file path')
        decodeing_file_label.setToolTip('The decoding file maps the output of a model into sematic labels.')
        horizontal_layout_decoding_file = QHBoxLayout()
        self.decoding_file_path_input = QLineEdit()
        self.decoding_file_path_input_btn = QPushButton('file')
        self.decoding_file_path_input_btn.clicked.connect(self.select_decoding_file)
        horizontal_layout_decoding_file.addWidget(self.decoding_file_path_input)
        horizontal_layout_decoding_file.addWidget(self.decoding_file_path_input_btn)
        return decodeing_file_label, horizontal_layout_decoding_file

    def init_output_dimensions_when_segmentation(self):
        output_shape = [0, 0, 0]
        output_dimen_label = QLabel('Output dimensions when segmentation')
        output_dimen_label.setToolTip('The dimension of the output when choose segmentation context')
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(QLabel('Class'))
        self.output_class = QLineEdit(str(output_shape[0]))
        horizontal_layout.addWidget(self.output_class)
        horizontal_layout.addWidget(QLabel('Width'))
        self.output_width = QLineEdit(str(output_shape[2]))
        horizontal_layout.addWidget(self.output_width)
        horizontal_layout.addWidget(QLabel('Height'))
        self.output_height = QLineEdit(str(output_shape[1]))
        horizontal_layout.addWidget(self.output_height)
        return output_dimen_label, horizontal_layout

    def init_output_is_channelfirst(self):
        output_channel_first_label = QLabel('Output is channel first when segmentation')
        output_channel_first_label.setToolTip('whether the output of models for segmentation tasks is width,'
                                              ' height, channel order or channel, width, height order.')
        self.output_is_whc_checkbox = QCheckBox()
        self.output_is_whc_checkbox.setChecked(False)
        return output_channel_first_label, self.output_is_whc_checkbox

    def init_add_softmax_activation(self):
        softmax_label = QLabel('Add softmax activation to output when segmentation')
        softmax_label.setToolTip('If some pretrained models do not have the softmax activation layer, So Here is'
                                 ' a option to add softmax activation layer which compute the prediction values.')
        self.apply_activation_on_output_checkbox = QCheckBox()
        self.apply_activation_on_output_checkbox.setChecked(True)
        return softmax_label, self.apply_activation_on_output_checkbox

    def init_apply_cancel_button(self):
        self.apply_btn = QPushButton('Apply', self)
        self.apply_btn.clicked.connect(self.apply_values)
        close_btn = QPushButton('Cancel', self)
        close_btn.clicked.connect(self.cancel)
        return self.apply_btn, close_btn

    def init_value(self):
        self.preprocessing_presets_selected(0)
        model = MVCMODEL.get_model()
        if model is None:
            return
        self.context_comboBox.setCurrentText(model.context)
        idx = CONTEXTS_PRESETS.index(model.context)
        self.context_selected(idx)
        self.image_channel.setText(str(model.input_shape[0]))
        self.image_width.setText(str(model.input_shape[2]))
        self.image_height.setText(str(model.input_shape[1]))
        idx = MODEL_PREPROCESS.get_all_preset_name().index(model.preprocess_setting_name)
        self.preprocessing_presets_selected(idx)
        check_state = QtCore.Qt.CheckState.Checked if model.should_normalize else QtCore.Qt.CheckState.Unchecked
        self.should_normalization_checkbox.setCheckState(check_state)
        check_state = QtCore.Qt.CheckState.Checked if model.channel_first else QtCore.Qt.CheckState.Unchecked
        self.channel_first_checkbox.setCheckState(check_state)
        check_state = QtCore.Qt.CheckState.Checked if model.image_is_rgb else QtCore.Qt.CheckState.Unchecked
        self.image_is_rgb_checkbox.setCheckState(check_state)
        self.decoding_file_path_input.setText(str(model.mapping_file_path))
        self.output_class.setText(str(model.output_shape[0]))
        self.output_width.setText(str(model.output_shape[2]))
        self.output_height.setText(str(model.output_shape[1]))
        check_state = QtCore.Qt.CheckState.Checked if model.output_is_whc else QtCore.Qt.CheckState.Unchecked
        self.output_is_whc_checkbox.setCheckState(check_state)
        check_state = QtCore.Qt.CheckState.Checked if model.add_output_activation else QtCore.Qt.CheckState.Unchecked
        self.apply_activation_on_output_checkbox.setCheckState(check_state)

    @QtCore.Slot(int)
    def preprocessing_presets_selected(self, index):
        preset_name = MODEL_PREPROCESS.get_all_preset_name()[index]
        presets_value = MODEL_PREPROCESS.get_preset_values(preset_name)
        if presets_value is None:
            return
        self.channel_0_in_mean.setText(str(presets_value['mean'][0]))
        self.channel_1_in_mean.setText(str(presets_value['mean'][1]))
        self.channel_2_in_mean.setText(str(presets_value['mean'][2]))
        self.channel_0_in_std.setText(str(presets_value["std"][0]))
        self.channel_1_in_std.setText(str(presets_value["std"][1]))
        self.channel_2_in_std.setText(str(presets_value["std"][2]))

    @QtCore.Slot(int)
    def context_selected(self, index):
        segmentation_context = is_segmentation(index)
        self.output_is_whc_checkbox.setEnabled(segmentation_context)
        self.apply_activation_on_output_checkbox.setEnabled(segmentation_context)
        self.output_class.setEnabled(segmentation_context)
        self.output_width.setEnabled(segmentation_context)
        self.output_height.setEnabled(segmentation_context)

    def apply_values(self):
        model = MVCMODEL.get_model()
        if model is None:
            return
        model.set_context(self.context_comboBox.currentIndex())
        try:
            import numpy as np
            model.input_shape = np.array([int(self.image_channel.text()),
                                          int(self.image_height.text()),
                                          int(self.image_width.text())])
            model.output_shape = np.array([int(self.output_class.text()),
                                          int(self.output_height.text()),
                                          int(self.output_width.text())])
            mean = [float(self.channel_0_in_mean.text()), float(self.channel_1_in_mean.text()), float(self.channel_2_in_mean.text())]
            standard_deviation = [float(self.channel_0_in_std.text()), float(self.channel_1_in_std.text()),float(self.channel_2_in_std.text())]
            model.preprocess_setting = {'mean': mean, 'std': standard_deviation}
        except ValueError:
            QMessageBox.critical(self.main_window,
                                 "Only float number input allowed.", QMessageBox.Ok)
        model.should_normalize = self.should_normalization_checkbox.isChecked()
        model.channel_first = self.channel_first_checkbox.isChecked()
        model.image_is_rgb = self.image_is_rgb_checkbox.isChecked()
        model.output_is_whc = self.output_is_whc_checkbox.isChecked()
        model.add_output_activation = self.apply_activation_on_output_checkbox.isChecked()
        model.mapping_file_path = self.decoding_file_path_input.text()
        self.close()

    def add_new_preprocessing_presets(self):
        name, selected_option = QInputDialog.getText(self, "Save as...", "Presets name: ")
        if selected_option and name is not None:
            presets_dict = dict()
            presets_dict["mean"] = [float(self.channel_0_in_mean.text()),
                                    float(self.channel_1_in_mean.text()),
                                    float(self.channel_2_in_mean.text()) ]
            presets_dict["std"] = [float(self.channel_0_in_std.text()),
                                   float(self.channel_1_in_std.text()),
                                   float(self.channel_2_in_std.text())]
            MODEL_PREPROCESS.append_presets(name, presets_dict)

    def select_decoding_file(self):
        results = QFileDialog.getOpenFileNames(self,
                                               "select decoding file",
                                               filter="Image Files (*.json)")
        # pylint disable=invalid-name
        for file_name in results[0]:
            try:
                self.decoding_file_path_input.setText(str(file_name))
            # pylint: disable=broad-except
            except Exception:
                QMessageBox.critical(self.image_window,
                                     "The selected file is not supported",
                                     "The file is corrupted or does not contain"
                                     " supported decoding information",
                                     QMessageBox.Ok)

    def cancel(self):
        self.setting_dialog_closed.emit()
        self.close()
