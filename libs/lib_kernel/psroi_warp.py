# -*- coding: utf-8 -*-

from __future__ import absolute_import,print_function,division

import tensorflow as tf

import tensorflow.contrib.slim as slim
import numpy as np


def psroi_warp(feature_maps, boxes, crop_size, num_spatial_bins):
    '''

    :param feature_maps:[N, H, W, C]
    :param boxes: [y1, x1, y2, x2] normalized coordinate
    :param crop_size: [9, 9]
    :param num_spatial_bins:[3, 3]
    :return:
    '''

    total_bins = 1
    bin_crop_size = []

    for (num_bins, crop_dim) in zip(num_spatial_bins, crop_size):
        if num_bins < 1:
            raise ValueError('num_spatial_bins should be >= 1')

        if crop_dim % num_bins != 0:
            raise ValueError('crop_size should be divisible by num_spatial_bins')

        total_bins *= num_bins
        bin_crop_size.append(crop_dim // num_bins)

    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    spatial_bins_y, spatial_bins_x = num_spatial_bins

    all_bin_boxes = []

    for bin_y in range(spatial_bins_y):
        step_y = (ymax - ymin)/spatial_bins_y
        for bin_x in range(spatial_bins_x):
            step_x = (xmax - xmin)/spatial_bins_x

            a_bin_boxes = [ymin+bin_y*step_y, xmin+bin_x*step_x,
                           ymax+bin_y*step_y, xmax+bin_x*step_x]

            a_bin_boxes = tf.stack(a_bin_boxes, axis=1)
            all_bin_boxes.append(a_bin_boxes)

    featuremap_splits = tf.split(feature_maps, num_or_size_splits=total_bins, axis=3)

    all_bin_crops = []
    for a_binboxes, a_group_featuremap in zip(all_bin_boxes, featuremap_splits):

        crop = tf.image.crop_and_resize(a_group_featuremap, boxes=a_binboxes,
                                        box_ind=tf.zeros(shape=[tf.shape(a_binboxes)[0], ],
                                                         dtype=tf.int32),
                                        crop_size=bin_crop_size)
        #  crop shape is : [N, bin_crop_size, bin_crop_size, featuremap_splits.shape[-1]]

        all_bin_crops.append(crop)

    final_results = tf.add_n(all_bin_crops) / len(all_bin_crops)  # [N, 1, 1, cls+1]

    final_results = tf.reduce_mean(final_results, [1, 2], keep_dims=False)  # [N, cls+1]

    return final_results


def position_sensitive_crop_regions(image,
                                    boxes,
                                    crop_size,
                                    num_spatial_bins,
                                    global_pool):
  """Position-sensitive crop and pool rectangular regions from a feature grid.
  The output crops are split into `spatial_bins_y` vertical bins
  and `spatial_bins_x` horizontal bins. For each intersection of a vertical
  and a horizontal bin the output values are gathered by performing
  `tf.image.crop_and_resize` (bilinear resampling) on a a separate subset of
  channels of the image. This reduces `depth` by a factor of
  `(spatial_bins_y * spatial_bins_x)`.
  When global_pool is True, this function implements a differentiable version
  of position-sensitive RoI pooling used in
  [R-FCN detection system](https://arxiv.org/abs/1605.06409).
  When global_pool is False, this function implements a differentiable version
  of position-sensitive assembling operation used in
  [instance FCN](https://arxiv.org/abs/1603.08678).
  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 3-D tensor of shape `[image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. Each box is specified in
      normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value
      of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so
      as the `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1] in image height coordinates. We do allow y1 > y2,
      in which case the sampled crop is an up-down flipped version of the
      original image. The width dimension is treated similarly.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    num_spatial_bins: A list of two integers `[spatial_bins_y, spatial_bins_x]`.
      Represents the number of position-sensitive bins in y and x directions.
      Both values should be >= 1. `crop_height` should be divisible by
      `spatial_bins_y`, and similarly for width.
      The number of image channels should be divisible by
      (spatial_bins_y * spatial_bins_x).
      Suggested value from R-FCN paper: [3, 3].
    global_pool: A boolean variable.
      If True, we perform average global pooling on the features assembled from
        the position-sensitive score maps.
      If False, we keep the position-pooled features without global pooling
        over the spatial coordinates.
      Note that using global_pool=True is equivalent to but more efficient than
        running the function with global_pool=False and then performing global
        average pooling.
  Returns:
    position_sensitive_features: A 4-D tensor of shape
      `[num_boxes, K, K, crop_channels]`,
      where `crop_channels = depth / (spatial_bins_y * spatial_bins_x)`,
      where K = 1 when global_pool is True (Average-pooled cropped regions),
      and K = crop_size when global_pool is False.
  Raises:
    ValueError: Raised in four situations:
      `num_spatial_bins` is not >= 1;
      `num_spatial_bins` does not divide `crop_size`;
      `(spatial_bins_y*spatial_bins_x)` does not divide `depth`;
      `bin_crop_size` is not square when global_pool=False due to the
        constraint in function space_to_depth.
  """

  image = tf.squeeze(image, axis=0)
  total_bins = 1
  bin_crop_size = []

  for (num_bins, crop_dim) in zip(num_spatial_bins, crop_size):
    if num_bins < 1:
      raise ValueError('num_spatial_bins should be >= 1')

    if crop_dim % num_bins != 0:
      raise ValueError('crop_size should be divisible by num_spatial_bins')

    total_bins *= num_bins
    bin_crop_size.append(crop_dim // num_bins)

  if not global_pool and bin_crop_size[0] != bin_crop_size[1]:
    raise ValueError('Only support square bin crop size for now.')

  ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
  spatial_bins_y, spatial_bins_x = num_spatial_bins

  # Split each box into spatial_bins_y * spatial_bins_x bins.
  position_sensitive_boxes = []
  for bin_y in range(spatial_bins_y):
    step_y = (ymax - ymin) / spatial_bins_y
    for bin_x in range(spatial_bins_x):
      step_x = (xmax - xmin) / spatial_bins_x
      box_coordinates = [ymin + bin_y * step_y,
                         xmin + bin_x * step_x,
                         ymin + (bin_y + 1) * step_y,
                         xmin + (bin_x + 1) * step_x,
                        ]
      position_sensitive_boxes.append(tf.stack(box_coordinates, axis=1))

  image_splits = tf.split(value=image, num_or_size_splits=total_bins, axis=2)

  image_crops = []
  for (split, box) in zip(image_splits, position_sensitive_boxes):
    if split.shape.is_fully_defined() and box.shape.is_fully_defined():
      crop = tf.squeeze(
          matmul_crop_and_resize(
              tf.expand_dims(split, axis=0), tf.expand_dims(box, axis=0),
              bin_crop_size),
          axis=0)
    else:
      crop = tf.image.crop_and_resize(
          tf.expand_dims(split, 0), box,
          tf.zeros([tf.shape(boxes)[0], ], dtype=tf.int32), bin_crop_size)
    image_crops.append(crop)

  if global_pool:
    # Average over all bins.
    position_sensitive_features = tf.add_n(image_crops) / len(image_crops)
    # Then average over spatial positions within the bins.
    position_sensitive_features = tf.reduce_mean(
        position_sensitive_features, [1, 2], keep_dims=True)
  else:
    # Reorder height/width to depth channel.
    block_size = bin_crop_size[0]
    if block_size >= 2:
      image_crops = [tf.space_to_depth(
          crop, block_size=block_size) for crop in image_crops]

    # Pack image_crops so that first dimension is for position-senstive boxes.
    position_sensitive_features = tf.stack(image_crops, axis=0)

    # Unroll the position-sensitive boxes to spatial positions.
    position_sensitive_features = tf.squeeze(
        tf.batch_to_space_nd(position_sensitive_features,
                             block_shape=[1] + num_spatial_bins,
                             crops=tf.zeros((3, 2), dtype=tf.int32)),
        squeeze_dims=[0])

    # Reorder back the depth channel.
    if block_size >= 2:
      position_sensitive_features = tf.depth_to_space(
          position_sensitive_features, block_size=block_size)

  return position_sensitive_features


def matmul_crop_and_resize(image, boxes, crop_size, scope=None):
  """Matrix multiplication based implementation of the crop and resize op.
  Extracts crops from the input image tensor and bilinearly resizes them
  (possibly with aspect ratio change) to a common output size specified by
  crop_size. This is more general than the crop_to_bounding_box op which
  extracts a fixed size slice from the input image and does not allow
  resizing or aspect ratio change.
  Returns a tensor with crops from the input image at positions defined at
  the bounding box locations in boxes. The cropped boxes are all resized
  (with bilinear interpolation) to a fixed size = `[crop_height, crop_width]`.
  The result is a 5-D tensor `[batch, num_boxes, crop_height, crop_width,
  depth]`.
  Running time complexity:
    O((# channels) * (# boxes) * (crop_size)^2 * M), where M is the number
  of pixels of the longer edge of the image.
  Note that this operation is meant to replicate the behavior of the standard
  tf.image.crop_and_resize operation but there are a few differences.
  Specifically:
    1) The extrapolation value (the values that are interpolated from outside
      the bounds of the image window) is always zero
    2) Only XLA supported operations are used (e.g., matrix multiplication).
    3) There is no `box_indices` argument --- to run this op on multiple images,
      one must currently call this op independently on each image.
    4) All shapes and the `crop_size` parameter are assumed to be statically
      defined.  Moreover, the number of boxes must be strictly nonzero.
  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, 'bfloat16', `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32` or 'bfloat16'.
      A 3-D tensor of shape `[batch, num_boxes, 4]`. The boxes are specified in
      normalized coordinates and are of the form `[y1, x1, y2, x2]`. A
      normalized coordinate value of `y` is mapped to the image coordinate at
      `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1] in image height coordinates.
      We do allow y1 > y2, in which case the sampled crop is an up-down flipped
      version of the original image. The width dimension is treated similarly.
      Normalized coordinates outside the `[0, 1]` range are allowed, in which
      case we use `extrapolation_value` to extrapolate the input image values.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    scope: A name for the operation (optional).
  Returns:
    A 5-D tensor of shape `[batch, num_boxes, crop_height, crop_width, depth]`
  Raises:
    ValueError: if image tensor does not have shape
      `[batch, image_height, image_width, depth]` and all dimensions statically
      defined.
    ValueError: if boxes tensor does not have shape `[batch, num_boxes, 4]`
      where num_boxes > 0.
    ValueError: if crop_size is not a list of two positive integers
  """
  img_shape = image.shape.as_list()
  boxes_shape = boxes.shape.as_list()
  _, img_height, img_width, _ = img_shape
  if not isinstance(crop_size, list) or len(crop_size) != 2:
    raise ValueError('`crop_size` must be a list of length 2')
  dimensions = img_shape + crop_size + boxes_shape
  if not all([isinstance(dim, int) for dim in dimensions]):
    raise ValueError('all input shapes must be statically defined')
  if len(boxes_shape) != 3 or boxes_shape[2] != 4:
    raise ValueError('`boxes` should have shape `[batch, num_boxes, 4]`')
  if len(img_shape) != 4:
    raise ValueError('image should have shape '
                     '`[batch, image_height, image_width, depth]`')
  num_crops = boxes_shape[0]
  if not num_crops > 0:
    raise ValueError('number of boxes must be > 0')
  if not (crop_size[0] > 0 and crop_size[1] > 0):
    raise ValueError('`crop_size` must be a list of two positive integers.')

  def _lin_space_weights(num, img_size):
    if num > 1:
      start_weights = tf.linspace(img_size - 1.0, 0.0, num)
      stop_weights = img_size - 1 - start_weights
    else:
      start_weights = tf.constant(num * [.5 * (img_size - 1)], dtype=tf.float32)
      stop_weights = tf.constant(num * [.5 * (img_size - 1)], dtype=tf.float32)
    return (start_weights, stop_weights)

  with tf.name_scope(scope, 'MatMulCropAndResize'):
    y1_weights, y2_weights = _lin_space_weights(crop_size[0], img_height)
    x1_weights, x2_weights = _lin_space_weights(crop_size[1], img_width)
    y1_weights = tf.cast(y1_weights, boxes.dtype)
    y2_weights = tf.cast(y2_weights, boxes.dtype)
    x1_weights = tf.cast(x1_weights, boxes.dtype)
    x2_weights = tf.cast(x2_weights, boxes.dtype)
    [y1, x1, y2, x2] = tf.unstack(boxes, axis=2)

    # Pixel centers of input image and grid points along height and width
    image_idx_h = tf.constant(
        np.reshape(np.arange(img_height), (1, 1, 1, img_height)),
        dtype=boxes.dtype)
    image_idx_w = tf.constant(
        np.reshape(np.arange(img_width), (1, 1, 1, img_width)),
        dtype=boxes.dtype)
    grid_pos_h = tf.expand_dims(
        tf.einsum('ab,c->abc', y1, y1_weights) + tf.einsum(
            'ab,c->abc', y2, y2_weights),
        axis=3)
    grid_pos_w = tf.expand_dims(
        tf.einsum('ab,c->abc', x1, x1_weights) + tf.einsum(
            'ab,c->abc', x2, x2_weights),
        axis=3)

    # Create kernel matrices of pairwise kernel evaluations between pixel
    # centers of image and grid points.
    kernel_h = tf.nn.relu(1 - tf.abs(image_idx_h - grid_pos_h))
    kernel_w = tf.nn.relu(1 - tf.abs(image_idx_w - grid_pos_w))

    # Compute matrix multiplication between the spatial dimensions of the image
    # and height-wise kernel using einsum.
    intermediate_image = tf.einsum('abci,aiop->abcop', kernel_h, image)
    # Compute matrix multiplication between the spatial dimensions of the
    # intermediate_image and width-wise kernel using einsum.
    return tf.einsum('abno,abcop->abcnp', kernel_w, intermediate_image)

