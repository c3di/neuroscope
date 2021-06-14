# -*-coding:utf-8-*-
import copy
import matplotlib.cm as mpl_color_map
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom


class ImageFilter:

    cmap_color_list = list(['seismic', 'viridis'])

    def __init__(self):
        self.cmap_color = 'seismic'
        self.alpha = 0.6

    def __call__(self, filter_name, src, **kwargs):
        return self.__getattribute__(filter_name)(src, **kwargs)

    # pylint: disable = unused-argument
    def positive(self, src, **kwargs):
        """return positive value of input"""
        return np.maximum(src, 0.0)

    # pylint: disable = unused-argument
    def negative(self, src, **kwargs):
        """return negative value of input"""
        return np.minimum(src, 0.0)

    # pylint: disable = unused-argument
    def normalize(self, src, **kwargs):
        """normalize to [0,1]"""
        src = src.astype('float')
        src = src - np.min(src)
        max_v = np.max(src)
        src = src / (max_v if max_v != 0 else 1.0)
        return src

    # pylint: disable = unused-argument
    def normalize_sign(self, src, **kwargs):
        """normalize to [-1, 1]"""
        src = src.astype('float')
        max_v = np.max(np.abs(src))
        src = src / (max_v if max_v != 0 else 1.0)
        return src

    # pylint: disable = unused-argument
    def normalize_on_each_image(self, src, **kwargs):
        """normalize to [-1, 1]"""
        src = src.astype('float')
        for i, array2d in enumerate(src):
            src[i, :, :] = self.normalize(array2d)
        return src

    # pylint: disable = unused-argument
    def normalize_sign_all(self, src, **kwargs):
        """normalize to [-1, 1]"""
        src = src.astype('float')
        for i, array2d in enumerate(src):
            src[i, :, :] = self.normalize_sign(array2d)
        return src

    # pylint: disable = unused-argument
    def heatmap(self, src, **kwargs):
        index = kwargs['item_index'] if kwargs['item_index'] != -1 else 0
        src = copy.deepcopy(src[index])
        background = kwargs['image'][0]
        src = zoom(src, (background.shape[0] / src.shape[0],
                                                        background.shape[1] / src.shape[1]))
        cmap = mpl_color_map.get_cmap(self.cmap_color)(self.normalize(src))
        heatmap = self.overlap_alpha_image(np.asarray(background),
                                           np.uint8(255 * cmap), self.alpha)
        return np.expand_dims(heatmap, axis=0)

    # pylint: disable = unused-argument
    def ground_truth(self, src, **kwargs):
        ground_truth = kwargs['image'][2]
        if kwargs['prediction_index'] != kwargs['output_classes']:
            one_hot_gt = np.eye(kwargs['output_classes'])[ground_truth.astype("uint8")][..., kwargs['prediction_index']] * 255
            heat_map = self.heatmap(np.expand_dims(one_hot_gt, axis=0), **{'image': kwargs['image'], 'item_index': 0})
        else:
            heat_map = self.normalize(ground_truth) * 255
            heat_map = np.expand_dims(heat_map, axis=0)

        return heat_map

    # pylint: disable = unused-argument
    def intersection(self, src, **kwargs):
        pred = Image.fromarray(kwargs['prediction'].result *kwargs['output_classes'])
        ground_truth = np.asarray(kwargs['image'][2])
        pred = pred.resize((ground_truth.shape[1], ground_truth.shape[0]))
        pred = np.array(pred)
        one_hot_gt = np.eye(kwargs['output_classes'])[ground_truth.astype("uint8")][..., kwargs['prediction_index']]
        one_hot_pred = np.eye(kwargs['output_classes'])[pred.astype("uint8")][..., kwargs['prediction_index']]
        intersection = np.logical_and(one_hot_gt, one_hot_pred)
        intersection = np.expand_dims(intersection/1, axis=0) * 255
        heat_map = self.heatmap(intersection, **{'image': kwargs['image'], 'item_index': 0})
        return heat_map

    # pylint: disable = unused-argument
    def union(self, src, **kwargs):
        pred = Image.fromarray(kwargs['prediction'].result *kwargs['output_classes'])
        ground_truth = np.asarray(kwargs['image'][2])
        pred = pred.resize((ground_truth.shape[1], ground_truth.shape[0]))
        pred = np.array(pred)
        one_hot_gt = np.eye(kwargs['output_classes'])[ground_truth.astype("uint8")][..., kwargs['prediction_index']]
        one_hot_pred = np.eye(kwargs['output_classes'])[pred.astype("uint8")][..., kwargs['prediction_index']]
        union = np.logical_or(one_hot_gt, one_hot_pred)
        union = np.expand_dims(union/1, axis=0) * 255
        heat_map = self.heatmap(union, **{'image': kwargs['image'], 'item_index': 0})
        return heat_map

    # pylint: disable = unused-argument
    def error(self, src, **kwargs):
        pred = Image.fromarray(kwargs['prediction'].result *kwargs['output_classes'])
        ground_truth = np.asarray(kwargs['image'][2])
        pred = pred.resize((ground_truth.shape[1], ground_truth.shape[0]))
        pred = np.array(pred)
        one_hot_gt = np.eye(kwargs['output_classes'])[ground_truth.astype("uint8")][..., kwargs['prediction_index']]
        one_hot_pred = np.eye(kwargs['output_classes'])[pred.astype("uint8")][..., kwargs['prediction_index']]
        error = np.logical_xor(one_hot_gt, one_hot_pred)
        error = np.expand_dims(error/1, axis=0) * 255
        heat_map = self.heatmap(error, **{'image': kwargs['image'], 'item_index': 0})
        return heat_map

    # pylint: disable = unused-argument
    def heatmap_neg(self, src, **kwargs):
        src = copy.deepcopy(src[0])
        cmap = mpl_color_map.get_cmap(self.cmap_color)(self.normalize(src))
        cmap[:, :, 3] = (self.normalize_sign(src) < 0).astype(np.uint8)
        heatmap = self.overlap_alpha_image(np.asarray(kwargs['image'][0]),
                                           np.uint8(255 * cmap), self.alpha)
        return np.expand_dims(heatmap, axis=0)

    # pylint: disable = unused-argument
    def heatmap_pos(self, src, **kwargs):
        src = copy.deepcopy(src[0])
        cmap = mpl_color_map.get_cmap(self.cmap_color)(self.normalize(src))
        cmap[:, :, 3] = (self.normalize_sign(src) > 0).astype(np.uint8)
        heatmap = self.overlap_alpha_image(np.asarray(kwargs['image'][0]),
                                           np.uint8(255 * cmap), self.alpha)
        return np.expand_dims(heatmap, axis=0)

    # pylint: disable = unused-argument
    def overlap_alpha_image(self, background_rgb, overlay_rgba,
                            alpha, gamma_factor=2.2):

        overlay_alpha = overlay_rgba[:, :, 3]
        overlay_alpha_3 = np.dstack((overlay_alpha, overlay_alpha,
                                     overlay_alpha))
        overlay_alpha_3 = self.resize(overlay_alpha_3, background_rgb.shape[1],
                                      background_rgb.shape[0])
        overlay_alpha_3 = overlay_alpha_3.astype(np.float) / 255. * alpha
        overlay_rgb_squared = self.resize(overlay_rgba[:, :, : 3],
                                          background_rgb.shape[1], background_rgb.shape[0])
        overlay_rgb_squared = np.float_power(overlay_rgb_squared
                                             .astype(np.float), gamma_factor)
        background_rgb_squared = np.float_power(
            background_rgb[:, :, :3].astype(np.float), gamma_factor)

        out_rgb_squared = overlay_rgb_squared * overlay_alpha_3 + \
                      background_rgb_squared * (1. - overlay_alpha_3)
        out_rgb = np.float_power(out_rgb_squared, 1. / gamma_factor)
        out_rgb = out_rgb.astype(np.uint8)
        return out_rgb

    # pylint: disable = unused-argument
    def color_mapping(self, src, **kwargs):
        return 255 * mpl_color_map.get_cmap(self.cmap_color)(self.normalize(src))

    # pylint: disable = unused-argument
    def color_mapping_on_each_image(self, src, **kwargs):
        return 255 * mpl_color_map.get_cmap(self.cmap_color)(self.normalize_on_each_image(src))

    # pylint: disable = unused-argument
    def pos_norm(self, src, **kwargs):
        return 255 * self.normalize((self.positive(src)))

    # pylint: disable = unused-argument
    def neg_norm(self, src, **kwargs):
        return 255 - 255 * self.normalize((self.negative(src)))

    # pylint: disable = unused-argument
    def normalized(self, src, **kwargs):
        return 255 * self.normalize(src)

    # pylint: disable = unused-argument
    def none(self, src, **kwargs):
        return src

    def resize(self, img, width=None, height=None):
        if width is None:
            return img
        img = Image.fromarray(img)
        img = img.resize((width, height))
        img = np.array(img)
        return img

IMAGE_FILTER = ImageFilter()
