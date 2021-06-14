import io
import numpy as np
import imageio
import matplotlib
import json
from PIL import Image
from matplotlib import pyplot as plt
from model.MVCModel import MVCMODEL
from .inspection import Inspection
from .predict_inspection import PredictInspection


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


class Confusion_matrix(Inspection):

    def __init__(self):
        super(Confusion_matrix, self).__init__()
        self.caption = "Confusion matrix"
        self.image_filters = ["none"]

    # pylint: disable=unused-argument
    def perform(self, input_image, layer=None, prediction=None, settings=None):
        input_images = MVCMODEL.images
        all_images = input_images.all_image_names()
        model = MVCMODEL.model
        output_classes = model.output_shape[0]
        confusions = []
        n_classes = output_classes
        for i in range(len(all_images)):
            image = input_images.get_image(i)
            ground_truth = image[2]
            if ground_truth is None:
                continue
            annotation_src = image[2]
            _, pred = PredictInspection().perform(image)
            pred = Image.fromarray(pred[-1].result * output_classes)
            ground_truth = annotation_src
            pred = pred.resize((ground_truth.shape[1], ground_truth.shape[0]))
            pred = np.array(pred)

            intersection_confusion = []
            one_hot_pred = np.eye(n_classes)[pred.astype("uint8")]
            scaled_annotation = ground_truth
            one_hot_annotation = np.eye(n_classes)[scaled_annotation.astype("uint8")]

            for cls in range(n_classes):
                one_hot_annotation_rolled = np.roll(one_hot_annotation, -cls, -1)
                intersection = np.logical_and(one_hot_pred, one_hot_annotation_rolled).sum((0, 1))
                intersection_confusion.append(intersection)
            intersection_confusion = np.array(intersection_confusion)
            for i in range(n_classes):
                intersection_confusion[:, i] = np.roll(intersection_confusion[:, i], i, 0)
            confusions.append(intersection_confusion)

        intersection_confusion = np.array(confusions).sum(0)
        prediction_sum = intersection_confusion.sum(-1)
        prediction_sum = np.resize(prediction_sum, (n_classes, n_classes))
        confusion_narmal = intersection_confusion / prediction_sum.T

        with open(model.mapping_file_path, "r") as file_reader:
            class_dict = json.load(file_reader)
        classes = range(0, n_classes)
        classes = [str(i) + class_dict[str(i)][0] for i in classes]
        plt.ioff()
        fig, ax = plt.subplots()
        plt.xlabel("True class")
        plt.ylabel("Predicted class")

        im, cbar = heatmap(confusion_narmal, classes, classes, ax=ax,
                           cmap="YlGn", cbarlabel="Confusion Matrix")
        annotate_heatmap(im, valfmt="{x:.2f}")

        fig.tight_layout()
        file = io.BytesIO()
        plt.savefig(file, format='PNG')
        file.seek(0)
        figure = imageio.imread(file)
        array = np.array(figure)[..., :3]
        return 'success', np.expand_dims(array, axis=0)
