import argparse
import chainer
import numpy as np
from chainercv.datasets import voc_bbox_label_names
from chainercv.evaluations import eval_detection_voc
from chainercv.utils import apply_prediction_to_iterator

import helper
import opt


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data_type', choices=opt.data_types, required=True)
    parser.add_argument('--det_type', choices=opt.detectors, required=True,
                        default='ssd300')
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

    dataset = helper.get_detection_dataset(args.data_type, 'test', args.root)

    iterator = chainer.iterators.SerialIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.predict, iterator, hook=helper.ProgressHook(len(dataset)))
    # delete unused iterator explicitly
    del imgs

    pred_bboxes, pred_labels, pred_scores = pred_values
    gt_bboxes, gt_labels = gt_values

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, use_07_metric=True)

    aps = result['ap']
    aps = aps[~np.isnan(aps)]

    print('')
    print('mAP: {:f}'.format(100.0 * result['map']))
    print(aps)


if __name__ == '__main__':
    main()
