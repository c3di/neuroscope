# -*-coding:utf-8-*-
import json
import os
PREPROCESSING_PRESET_FILE_PATH = "example/preprocessing_presets.json"


class Preprocessing:
    def __init__(self):
        self.presets_dict = dict()
        if os.path.exists(PREPROCESSING_PRESET_FILE_PATH):
            self.load_preset_from_file(PREPROCESSING_PRESET_FILE_PATH)

    def load_preset_from_file(self, preset_file_path):
        with open(preset_file_path) as presets_reader:
            self.presets_dict = json.load(presets_reader)

    def append_presets(self, name, value):
        self.presets_dict[name] = value
        with open(PREPROCESSING_PRESET_FILE_PATH, "w") as outfile:
            json.dump(self.presets_dict, outfile)

    def get_preset_values(self, preset_name):
        if preset_name not in self.presets_dict:
            raise Exception("This preset is not exist.")
        return self.presets_dict[preset_name]

    def get_all_preset_name(self):
        return list(self.presets_dict.keys())

    def __call__(self, needed_preprocess_input, mean, standard_deviation,
                 should_normalize=False, channel_first=False, is_rgb=True):
        """Preprocesses a Numpy array encoding a batch of images.
         data_format='channels_last'
        # Arguments
            input_img: Input array, 3D or 4D.
            data_format: Data format of the image array.
            mode: "tf" or "torch".
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
                - torch: will scale pixels between 0 and 1 and then
                    will normalize each channel with respect to the
                    ImageNet dataset.

        # Returns
            Preprocessed Numpy array.
        """
        import numpy as np
        if not is_rgb:
            # rgb_to_bgr
            needed_preprocess_input = needed_preprocess_input[:, :, ::-1]
        needed_preprocess_input = np.expand_dims(needed_preprocess_input, axis=0)
        needed_preprocess_input = needed_preprocess_input.astype(np.float64)
        if not needed_preprocess_input.flags['WRITEABLE']:
            needed_preprocess_input.setflags(write=True)
        num_of_channels = needed_preprocess_input.shape[3]
        for channel in range(num_of_channels):
            if should_normalize:
                needed_preprocess_input[..., channel] /= 255.0
            needed_preprocess_input[..., channel] -= mean[channel]
            if standard_deviation[channel] != 0:
                needed_preprocess_input[..., channel] /= standard_deviation[channel]
        if channel_first:
            needed_preprocess_input = needed_preprocess_input.transpose(0, 3, 1, 2)
        return needed_preprocess_input


MODEL_PREPROCESS = Preprocessing()
