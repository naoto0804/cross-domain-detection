# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

import os

import cv2

from lib.voc_io import PascalVocWriter


class LabelFileError(Exception):
    pass


class LabelFile(object):
    # It might be changed as window creates
    suffix = '.lif'

    def __init__(self, filename, imagePath, classes):
        assert(os.path.exists(imagePath))
        self.shapes = ()
        self.classes = classes
        self.imagePath = imagePath
        self.filename = filename
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        self.imageShape = image.shape

    def savePascalVocFormat(self, dets):
        imgFolderPath = os.path.dirname(self.imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(self.imagePath)
        imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        writer = PascalVocWriter(imgFolderName, imgFileNameWithoutExt,
                                 self.imageShape, localImgPath=self.imagePath)

        for cls in self.classes:
            for bbox in dets[cls]:
                bbox = self.prettifyBndBox(bbox)
                writer.addBndBox(bbox[0], bbox[1], bbox[2], bbox[3], cls)

        writer.save(targetFile=self.filename)
        return

    def toggleVerify(self):
        self.verified = not self.verified

    def prettifyBndBox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        height, width, _ = self.imageShape
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        if xmax > width:
            xmax = width

        if ymax > height:
            ymax = height

        return (int(xmin), int(ymin), int(xmax), int(ymax))

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix


if __name__ == '__main__':
    imgPath = "/home/inoue/data/VOCdevkit/CLIPART/JPEGImages/0001.jpg"

    filename = 'test.xml'
    dets = [{'bbox': (3, 4, 5, 6), 'label': 'car'},
              {'bbox': (5, 4, 5, 1), 'label': 'dog'}]
    # print(labeler.convertPoints2BndBox([(-1, 6), (5, 3)]))
    labeler = LabelFile(filename, imgPath)
    labeler.savePascalVocFormat(dets)