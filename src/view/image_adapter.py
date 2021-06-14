import numpy as np
import qimage2ndarray


def convert_float_image(image):
    _min = image.min()
    _max = image.max()
    _range = _max - _min
    transformed_image = image - _min
    transformed_image = transformed_image * _range
    transformed_image = transformed_image * 255
    return transformed_image.astype(int)


def numpy_to_qimage(image):
    if image is None:
        return None

    if image.dtype == np.uint8:
        return qimage2ndarray.array2qimage(image)

    if image.dtype == np.float32:
        image = convert_float_image(image)
        return qimage2ndarray.array2qimage(image)

    raise Exception("array of type {} cannot be displayed as image".format(image.dtype))
