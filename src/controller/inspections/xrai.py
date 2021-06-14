import numpy as np
from skimage import segmentation
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.transform import resize


_FELZENSZWALB_SCALE_VALUES = [50, 100, 150, 250, 500, 1200]
_FELZENSZWALB_SIGMA_VALUES = [0.8]
_FELZENSZWALB_IM_RESIZE = (224, 224)
_FELZENSZWALB_IM_VALUE_RANGE = [-1.0, 1.0]
_FELZENSZWALB_MIN_SEGMENT_SIZE = 150


def _normalize_image(im, value_range, resize_shape=None):
    """Normalize an image by resizing it and rescaling its values

  Args:
      im: Input image.
      value_range: [min_value, max_value]
      resize_shape: New image shape. Defaults to None.

  Returns:
      Resized and rescaled image.
  """
    im_max = np.max(im)
    im_min = np.min(im)
    im = (im - im_min) / (im_max - im_min)
    im = im * (value_range[1] - value_range[0]) + value_range[0]
    if resize_shape is not None:
        im = resize(im,
                    resize_shape,
                    order=3,
                    mode='constant',
                    preserve_range=True,
                    anti_aliasing=True)
    return im


def _get_segments_felzenszwalb(im,
                               resize_image=True,
                               scale_range=None,
                               dilation_rad=5):
    """Compute image segments based on Felzenszwalb's algorithm.

  Efficient graph-based image segmentation, Felzenszwalb, P.F.
  and Huttenlocher, D.P. International Journal of Computer Vision, 2004

  Args:
    im: Input image.
    resize_image: If True, the image is resized to 224,224 for the segmentation
                  purposes. The resulting segments are rescaled back to match
                  the original image size. It is done for consistency w.r.t.
                  segmentation parameter range. Defaults to True.
    scale_range:  Range of image values to use for segmentation algorithm.
                  Segmentation algorithm is sensitive to the input image
                  values, therefore we need to be consistent with the range
                  for all images. If None is passed, the range is scaled to
                  [-1.0, 1.0]. Defaults to None.
    dilation_rad: Sets how much each segment is dilated to include edges,
                  larger values cause more blobby segments, smaller values
                  get sharper areas. Defaults to 5.
  Returns:
      masks: A list of boolean masks as np.ndarrays if size HxW for im size of
             HxWxC.
  """

    # TODO (tolgab) Set this to default float range of 0.0 - 1.0 and tune
    # parameters for that
    if scale_range is None:
        scale_range = _FELZENSZWALB_IM_VALUE_RANGE
    # Normalize image value range and size
    original_shape = im.shape[:2]
    # TODO (tolgab) This resize is unnecessary with more intelligent param range
    # selection
    if resize_image:
        im = _normalize_image(im, scale_range, _FELZENSZWALB_IM_RESIZE)
    else:
        im = _normalize_image(im, scale_range)
    segs = []
    for scale in _FELZENSZWALB_SCALE_VALUES:
        for sigma in _FELZENSZWALB_SIGMA_VALUES:
            seg = segmentation.felzenszwalb(im,
                                            scale=scale,
                                            sigma=sigma,
                                            min_size=_FELZENSZWALB_MIN_SEGMENT_SIZE)
            if resize_image:
                seg = resize(seg,
                             original_shape,
                             order=0,
                             preserve_range=True,
                             mode='constant',
                             anti_aliasing=False).astype(np.int)
            segs.append(seg)
    masks = _unpack_segs_to_masks(segs)
    if dilation_rad:
        selem = disk(dilation_rad)
        masks = [dilation(mask, selem=selem) for mask in masks]
    return masks


def _attr_aggregation_max(attr, axis=-1):
    return attr.max(axis=axis)


def _gain_density(mask1, attr, mask2=None):
    # Compute the attr density over mask1. If mask2 is specified, compute density
    # for mask1 \ mask2
    if mask2 is None:
        added_mask = mask1
    else:
        added_mask = _get_diff_mask(mask1, mask2)
    if not np.any(added_mask):
        return -np.inf
    else:
        return attr[added_mask].mean()


def _get_diff_mask(add_mask, base_mask):
    return np.logical_and(add_mask, np.logical_not(base_mask))


def _get_diff_cnt(add_mask, base_mask):
    return np.sum(_get_diff_mask(add_mask, base_mask))


def _unpack_segs_to_masks(segs):
    masks = []
    for seg in segs:
        for l in range(seg.min(), seg.max() + 1):
            masks.append(seg == l)
    return masks


def _xrai(attr,
          segs,
          gain_fun=_gain_density,
          area_perc_th=1.0,
          min_pixel_diff=50,
          integer_segments=True):
    """Run XRAI saliency given attributions and segments.

    Args:
        attr: Source attributions for XRAI. XRAI attributions will be same size
              as the input attr.
        segs: Input segments as a list of boolean masks. XRAI uses these to
              compute attribution sums.
        gain_fun: The function that computes XRAI area attribution from source
                  attributions. Defaults to _gain_density, which calculates the
                  density of attributions in a mask.
        area_perc_th: The saliency map is computed to cover area_perc_th of
                      the image. Lower values will run faster, but produce
                      uncomputed areas in the image that will be filled to
                      satisfy completeness. Defaults to 1.0.
        min_pixel_diff: Do not consider masks that have difference less than
                        this number compared to the current mask. Set it to 1
                        to remove masks that completely overlap with the
                        current mask.
        integer_segments: See XRAIParameters. Defaults to True.

    Returns:
        tuple: saliency heatmap and list of masks or an integer image with
               area ranks depending on the parameter integer_segments.
    """
    output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

    n_masks = len(segs)
    current_area_perc = 0.0
    current_mask = np.zeros(attr.shape, dtype=bool)

    masks_trace = []
    remaining_masks = {ind: mask for ind, mask in enumerate(segs)}
    output_map = []
    added_masks_cnt = 1
    # While the mask area is less than area_th and remaining_masks is not empty
    while current_area_perc <= area_perc_th:
        best_gain = -np.inf
        best_key = None
        remove_key_queue = []
        for mask_key in remaining_masks:
            mask = remaining_masks[mask_key]
            # If mask does not add more than min_pixel_diff to current mask, remove
            mask_pixel_diff = _get_diff_cnt(mask, current_mask)
            if mask_pixel_diff < min_pixel_diff:
                # remove_key_queue.append(mask_key)
                # continue
                skiped = 1
            gain = gain_fun(mask, attr, mask2=current_mask)
            if gain > best_gain:
                best_gain = gain
                best_key = mask_key
        for key in remove_key_queue:
            del remaining_masks[key]
        if len(remaining_masks) == 0:
            break
        if best_key is None:
            best_key = mask_key
            best_gain = 0
        added_mask = remaining_masks[best_key]

        mask_diff = _get_diff_mask(added_mask, current_mask)
        masks_trace.append((mask_diff, best_gain))

        current_mask = np.logical_or(current_mask, added_mask)
        current_area_perc = np.mean(current_mask)
        output_attr[mask_diff] = best_gain
        output_map.append((best_key, best_gain))
        del remaining_masks[best_key]  # delete used key

    uncomputed_mask = output_attr == -np.inf
    # Assign the uncomputed areas a value such that sum is same as ig
    output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)
    masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
    if np.any(uncomputed_mask):
        masks_trace.append(uncomputed_mask)
    if integer_segments:
        attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
        for i, mask in enumerate(masks_trace):
            attr_ranks[mask] = i + 1
        return output_attr, attr_ranks, output_map
    else:
        return output_attr, masks_trace


def GetMaskWithDetails(
        x_value,
        feed_dict={},
        baselines=None,
        segments=None,
        base_attribution=None,
        extra_parameters=None,
        just_output_maps=False
        ):
    """Applies XRAI method on an input image and returns the result saliency
    heatmap along with other detailed information.


    Args:
        just_output_maps: uses just network segmentation results
        x_value: input value, not batched.
        feed_dict: feed dictionary to pass to the TF session.run() call.
                   Defaults to {}.
        baselines: a list of baselines to use for calculating
                   Integrated Gradients attribution. Every baseline in
                   the list should have the same dimensions as the
                   input. If the value is not set then the algorithm
                   will make the best effort to select default
                   baselines. Defaults to None.
        segments: the list of precalculated image segments that should
                  be passed to XRAI. Each element of the list is an
                  [N,M] boolean array, where NxM are the image
                  dimensions. Each elemeent on the list contains exactly the
                  mask that corresponds to one segment. If the value is None,
                  Felzenszwalb's segmentation algorithm will be applied.
                  Defaults to None.
        base_attribution: an optional pre-calculated base attribution that XRAI
                          should use. The shape of the parameter should match
                          the shape of `x_value`. If the value is None, the
                          method calculates Integrated Gradients attribution and
                          uses it.
        extra_parameters: an XRAIParameters object that specifies
                          additional parameters for the XRAI saliency
                          method. If it is None, an XRAIParameters object
                          will be created with default parameters. See
                          XRAIParameters for more details.

    Raises:
        ValueError: If algorithm type is unknown (not full or fast).
                    If the shape of `base_attribution` dosn't match the shape of `x_value`.

    Returns:
        XRAIOutput: an object that contains the output of the XRAI algorithm.

    TODO(tolgab) Add output_selector functionality from XRAI API doc
    """
    if extra_parameters is None:
        algorithm = 'full'
    # Check the shape of base_attribution.
    if base_attribution is not None:
        if not isinstance(base_attribution, np.ndarray):
            base_attribution = np.array(base_attribution)
        if base_attribution.shape != x_value.shape:
            raise ValueError(
                'The base attribution shape should be the same as the shape of '
                '`x_value`. Expected {}, got {}'.format(
                    x_value.shape, base_attribution.shape))

    # Calculate IG attribution if not provided by the caller.
    else:
        x_baselines = None
        attrs = base_attribution
    attr = base_attribution
    # Merge attribution channels for XRAI input
    attr = _attr_aggregation_max(attr)
    if segments is not None:
        if just_output_maps:
            segs = segments
        else:
            segs = _get_segments_felzenszwalb(x_value)
            segs = np.concatenate((segs, segments), 0)
    else:
        segs = _get_segments_felzenszwalb(x_value)
    if algorithm == 'full':
        attr_map, attr_data, output_map = _xrai(
            attr=attr,
            segs=segs,
            area_perc_th=1.0,
            min_pixel_diff=50,
            gain_fun=_gain_density,
            integer_segments=True)
    else:
        raise ValueError('Unknown algorithm type: {}'.format(
            extra_parameters.algorithm))
    results = (attr_map, output_map)
    return results
