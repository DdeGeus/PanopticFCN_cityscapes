# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implement many useful :class:`Augmentation`.
"""
import numpy as np
from typing import Any, List, Optional, Tuple, Union

from fvcore.transforms.transform import (
    CropTransform,
    Transform,
    TransformList
)

from detectron2.data.transforms.augmentation import Augmentation, AugmentationList
from detectron2.data.transforms.augmentation_impl import RandomCrop

__all__ = [
    "RandomCropWithInstance",
    "AugInput"
]


def _check_img_dtype(img):
  assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(
    type(img)
  )
  assert not isinstance(img.dtype, np.integer) or (
      img.dtype == np.uint8
  ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(
    img.dtype
  )
  assert img.ndim in [2, 3], img.ndim


class RandomCropWithInstance(Augmentation):
  """
  Similar to :class:`RandomCrop`, but find a cropping window such that no single category
  occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
  truth, which can cause unstability in training. The function attempts to find such a valid
  cropping window for at most 10 times.
  """

  def __init__(
      self,
      crop_type: str,
      crop_size,
      min_inst_area: int = None,
  ):
    """
    Args:
        crop_type, crop_size: same as in :class:`RandomCrop`
        min_inst_area: the minimum amount of things instance pixels that should be present in the crop
    """
    self.crop_aug = RandomCrop(crop_type, crop_size)
    self._init(locals())

  def get_transform(self, image, inst_map):
    """
    Args:
        image: image
        inst_map: binary map of things instance locations in image, dtype int32. 1 if instance is present.
    """
    if self.min_inst_area <= 0:
      return self.crop_aug.get_transform(image)
    else:
      h, w = inst_map.shape
      for i in range(10):
        crop_size = self.crop_aug.get_crop_size((h, w))
        y0 = np.random.randint(h - crop_size[0] + 1)
        x0 = np.random.randint(w - crop_size[1] + 1)
        inst_map_tmp = inst_map[y0: y0 + crop_size[0], x0: x0 + crop_size[1]]
        cnt = np.sum(inst_map_tmp)
        if cnt > self.min_inst_area:
          break
      crop_tfm = CropTransform(x0, y0, crop_size[1], crop_size[0])
      return crop_tfm


class AugInput:
  """
  Input that can be used with :meth:`Augmentation.__call__`.
  This is a standard implementation for the majority of use cases.
  This class provides the standard attributes **"image", "boxes", "sem_seg"**
  defined in :meth:`__init__` and they may be needed by different augmentations.
  Most augmentation policies do not need attributes beyond these three.

  After applying augmentations to these attributes (using :meth:`AugInput.transform`),
  the returned transforms can then be used to transform other data structures that users have.

  Examples:
  ::
      input = AugInput(image, boxes=boxes)
      tfms = augmentation(input)
      transformed_image = input.image
      transformed_boxes = input.boxes
      transformed_other_data = tfms.apply_other(other_data)

  An extended project that works with new data types may implement augmentation policies
  that need other inputs. An algorithm may need to transform inputs in a way different
  from the standard approach defined in this class. In those rare situations, users can
  implement a class similar to this class, that satify the following condition:

  * The input must provide access to these data in the form of attribute access
    (``getattr``).  For example, if an :class:`Augmentation` to be applied needs "image"
    and "sem_seg" arguments, its input must have the attribute "image" and "sem_seg".
  * The input must have a ``transform(tfm: Transform) -> None`` method which
    in-place transforms all its attributes.
  """

  # TODO maybe should support more builtin data types here
  def __init__(
      self,
      image: np.ndarray,
      *,
      boxes: Optional[np.ndarray] = None,
      sem_seg: Optional[np.ndarray] = None,
      inst_map: Optional[np.ndarray] = None,
  ):
    """
    Args:
        image (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
            floating point in range [0, 1] or [0, 255]. The meaning of C is up
            to users.
        boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
        sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
            is an integer label of pixel.
    """
    _check_img_dtype(image)
    self.image = image
    self.boxes = boxes
    self.sem_seg = sem_seg
    self.inst_map = inst_map

  def transform(self, tfm: Transform) -> None:
    """
    In-place transform all attributes of this class.

    By "in-place", it means after calling this method, accessing an attribute such
    as ``self.image`` will return transformed data.
    """
    self.image = tfm.apply_image(self.image)
    if self.boxes is not None:
      self.boxes = tfm.apply_box(self.boxes)
    if self.sem_seg is not None:
      self.sem_seg = tfm.apply_segmentation(self.sem_seg)
    if self.inst_map is not None:
      self.inst_map = tfm.apply_segmentation(self.inst_map)

  def apply_augmentations(
      self, augmentations: List[Union[Augmentation, Transform]]
  ) -> TransformList:
    """
    Equivalent of ``AugmentationList(augmentations)(self)``
    """
    return AugmentationList(augmentations)(self)
