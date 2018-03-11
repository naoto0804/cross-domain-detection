import os
import xml.etree.ElementTree as ET

import chainer
import numpy as np
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image

from opt import bam_contents_classes


class BaseDetectionDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, subset, use_difficult, return_difficult):
        self.root = root
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.imgset_dir = os.path.join(root, 'ImageSets/Main')
        self.ann_dir = os.path.join(root, 'Annotations')
        id_list_file = os.path.join(
            self.imgset_dir, '{:s}.txt'.format(subset))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.subset = subset
        self.labels = None  # for network
        self.actual_labels = None  # for visualization

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.ann_dir, id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        objs = anno.findall('object')

        for obj in objs:
            # If not using difficult split, and the object is
            # difficult, skip it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            bndbox_anno = obj.find('bndbox')
            name = obj.find('name').text.lower().strip()
            label.append(self.labels.index(name))
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            difficult.append(int(obj.find('difficult').text))

        try:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            difficult = np.array(difficult, dtype=np.bool)
        except ValueError:
            bbox = np.empty((0, 4), dtype=np.float32)
            label = np.empty((0,), dtype=np.int32)
            difficult = np.empty((0,), dtype=np.bool)

        # Load a image
        img_file = os.path.join(self.img_dir, id_ + '.jpg')
        img = read_image(img_file, color=True)
        if self.return_difficult:
            return img, bbox, label, difficult
        return img, bbox, label


class VOCDataset(BaseDetectionDataset):
    def __init__(self, root, subset, use_difficult=False,
                 return_difficult=False):
        super(VOCDataset, self).__init__(root, subset, use_difficult,
                                         return_difficult)
        self.labels = voc_utils.voc_bbox_label_names
        self.actual_labels = voc_utils.voc_bbox_label_names


class ClipArtDataset(BaseDetectionDataset):
    def __init__(self, root, subset, use_difficult=False,
                 return_difficult=False):
        super(ClipArtDataset, self).__init__(root, subset, use_difficult,
                                             return_difficult)
        self.labels = voc_utils.voc_bbox_label_names
        self.actual_labels = voc_utils.voc_bbox_label_names


class BAMDataset(BaseDetectionDataset):
    def __init__(self, root, subset, use_difficult=False,
                 return_difficult=False):
        super(BAMDataset, self).__init__(root, subset, use_difficult,
                                         return_difficult)
        self.labels = voc_utils.voc_bbox_label_names
        self.actual_labels = bam_contents_classes
