# -*-coding:utf-8-*-

import imageio
import numpy
from PySide2.QtCore import QObject, Signal


class InputImages(QObject):

    added = Signal(str, int)
    gt_added = Signal(int, str)
    removed = Signal(int)

    def __init__(self):
        super(InputImages, self).__init__()
        self.images = []
        self.default_image = imageio.imread("resources/images/emptyImage.png")

    def __len__(self):
        return len(self.images)

    def add_image(self, image, name):
        image = numpy.array(image)
        self.images.append((image, name, None))
        self.added.emit(name, len(self.images))
        return len(self.images)

    def add_ground_truth(self, image_index, ground_truth, gt_file_name):
        if len(ground_truth.shape) > 2:
            ground_truth = numpy.array(ground_truth)[..., 0]
        else:
            ground_truth = numpy.array(ground_truth)
        self.images[image_index] = (self.images[image_index][0],
                                    self.images[image_index][1], ground_truth)
        self.gt_added.emit(image_index, gt_file_name)
        return len(self.images)

    def set_ground_truth_directory(self, dir_path):
        import os
        file_list = []
        directory = os.fsencode(dir_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".png"):
                file_list.append(filename)
        all_image_names = self.all_image_names()
        for filename in file_list:
            if filename in all_image_names:
                image_index = all_image_names.index(filename)
                gt_full_path = os.path.join(dir_path, filename)
                ground_truth = imageio.imread(gt_full_path)
                self.add_ground_truth(image_index, ground_truth, gt_full_path)

    def all_image_names(self):
        all_names = []
        for (_, name, _) in self.images:
            all_names.append(name)
        return all_names

    def get_image(self, index):
        if not self.images:
            return None
        return self.images[index]

    def get_all_images(self):
        if not self.images:
            return self.default_image
        return self.images

    def get_image_name(self, index):
        if not self.images:
            return None
        return self.images[index][1]

    def remove_image(self, index):
        del self.images[index]
        self.removed.emit(index)
        return len(self.images)

    def remove_all_image(self):
        size = len(self.images)
        for i in range(size - 1, -1, -1):
            self.remove_image(i)
