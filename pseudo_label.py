from collections import defaultdict

import argparse
import chainer
import numpy as np
import os
from chainercv.datasets import voc_bbox_label_names
from chainercv.utils import apply_prediction_to_iterator

import helper
import opt
from lib.label_file import LabelFile


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data_type', choices=opt.data_types, required=True)
    parser.add_argument('--det_type', choices=opt.detectors, required=True,
                        default='ssd300')
    parser.add_argument('--result', required=True)
    parser.add_argument('--load', help='load original trained model')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    args = parser.parse_args()

    model_args = {'n_fg_class': len(voc_bbox_label_names),
                  'pretrained_model': 'voc0712'}
    model = helper.get_detector(args.det_type, model_args)

    if args.load:
        chainer.serializers.load_npz(args.load, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    model.use_preset('evaluate')

    dataset = helper.get_detection_dataset(args.data_type, 'train', args.root)

    iterator = chainer.iterators.SerialIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.predict, iterator, hook=helper.ProgressHook(len(dataset)))
    # delete unused iterator explicitly
    del imgs

    pred_bboxes, pred_labels, pred_scores = pred_values
    _, gt_labels = gt_values

    ids = []

    for i, (pred_b, pred_l, pred_s, gt_l) in enumerate(
            zip(pred_bboxes, pred_labels, pred_scores, gt_labels)):

        labels = dataset.labels
        proper_dets = defaultdict(list)
        name = dataset.ids[i]
        cnt = 0

        gt_l = set(gt_l)
        for l in set(gt_l):
            cnt += 1
            class_indices = np.where(pred_l == l)[0]
            if len(class_indices) == 0:
                continue
            scores = pred_s[class_indices]
            ind = class_indices[np.argsort(scores)[::-1][0]]  # top1
            assert (l == pred_l[ind])
            proper_dets[labels[l]].append(pred_b[ind][[1, 0, 3, 2]])

        if cnt == 0:
            continue

        ids.append(dataset.ids[i] + '\n')
        filename = os.path.join(args.result, 'Annotations', name + '.xml')
        img_path = os.path.join(args.result, 'JPEGImages', name + '.jpg')
        labeler = LabelFile(filename, img_path, dataset.actual_labels)
        labeler.savePascalVocFormat(proper_dets)

    txt = 'ImageSets/Main/train.txt'
    with open(os.path.join(args.result, txt), 'w') as f:
        f.writelines(ids)
    print('Saved to {:s}'.format(args.result))


if __name__ == '__main__':
    main()
