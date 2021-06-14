# -*-coding:utf-8-*-
import json
import numpy as np
from PIL import Image


class PredictionResult:
    def __init__(self, class_id=None, name=None, result=None, index=None):
        self.class_id = class_id
        self.name = name
        self.result = result
        self.index = index


class DecodePredictions:
    def __call__(self, model, input_image, predictions, top=5):
        if model.context == "Classification":
            return self.classification_decode_predictions(model, predictions, top)
        elif model.context == "Segmentation":
            return self.segmentation_decode_predictions(model, predictions, input_image)
        else:
            raise Exception("Decode is not supported")

    def segmentation_decode_predictions(self, model=None, predictions=None, input_image=None):
        with open(model.mapping_file_path, "r") as file_reader:
            class_index = json.load(file_reader)
        n_classes, _, _ = model.output_shape
        predictions = self.reshape_segmentation_predictions(model, predictions)
        segmented_image, iou = self.get_segmented_image_iou(input_image, n_classes, predictions)
        segmented_image = np.expand_dims(segmented_image, 0)
        predictions = np.concatenate((predictions, segmented_image), 0)
        prediction_result = self.find_segmentation_prediction_results(class_index, iou, predictions)
        return prediction_result

    def get_segmented_image_iou(self, input_image, n_classes, predictions):
        if predictions.shape[0] == 1:
            segmented_image = predictions.round()[0] / n_classes
        else:
            segmented_image = predictions.argmax(axis=0) / n_classes
        if input_image[2] is not None:
            iou = self.intersection_over_union(segmented_image, input_image[2], n_classes)
        else:
            iou = np.zeros(n_classes + 1)
        return segmented_image, iou

    def find_segmentation_prediction_results(self, class_index, iou, predictions):
        results = list()
        for i, prediction in enumerate(predictions):
            name = str(i) + ' ' + class_index[str(i)][0] + ' ' + "{0:.2f}".format(iou[i])
            results.append(PredictionResult(name=name,
                                            result=prediction,
                                            index=i))
        return results

    def reshape_segmentation_predictions(self, model, predictions):
        predictions = np.maximum(predictions, 0)
        n_classes, output_height, output_width = model.output_shape
        if model.output_is_whc:
            predictions = predictions.reshape(output_height, output_width, n_classes)
            predictions = np.transpose(predictions, (2, 0, 1))
        else:
            predictions = predictions.reshape(n_classes, output_height, output_width)
        return predictions

    def classification_decode_predictions(self, model=None, predictions=None, top=5):
        with open(model.mapping_file_path, "r") as file_reader:
            class_index = json.load(file_reader)
        if class_index is None:
            return []
        if len(predictions.shape) > 1:
            predictions = predictions[0]
        prediction_result = self.find_top_n_prediction_results(predictions, class_index, top)
        return prediction_result

    def find_top_n_prediction_results(self, predictions, class_index, top_n):
        results = list()
        top_indices = predictions.argsort()[-top_n:][::-1]
        for index in top_indices:
            results.append(PredictionResult(class_id=class_index[str(index)][0],
                                            name=class_index[str(index)][1] + " " +
                                            str(np.around(predictions[index], decimals=3)),
                                            result=predictions[index],
                                            index=index))
        return results

    def intersection_over_union(self, prediction, ground_truth, n_classes):
        prediction = Image.fromarray(prediction.astype('uint8'))
        prediction = prediction.resize((ground_truth.shape[1], ground_truth.shape[0]))
        prediction = np.array(prediction)
        ground_truth = np.minimum(ground_truth, n_classes-1)
        one_hot_prediction = np.eye(n_classes)[prediction.astype("uint8")]
        one_hot_annotation = np.eye(n_classes)[ground_truth.astype("uint8")]
        intersection = np.logical_and(one_hot_prediction, one_hot_annotation).sum((0, 1))
        union = np.logical_or(one_hot_prediction, one_hot_annotation).sum((0, 1))
        iou = (intersection/union)
        return np.append(iou, [[iou.mean()]])


DECODE_PREDICTIONS = DecodePredictions()
