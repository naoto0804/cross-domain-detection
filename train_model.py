#! /usr/bin/env python3

import argparse
import chainer
import chainercv.links
import copy
import numpy as np
import os
from chainer import serializers
from chainer import training
from chainer.datasets import TransformDataset
from chainer.iterators import MultiprocessIterator
from chainer.iterators import SerialIterator
from chainer.optimizer import WeightDecay
from chainer.training import extensions
from chainercv import transforms
from chainercv.datasets import voc_bbox_label_names
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

import opt
from helper import get_detection_dataset


class ConcatenatedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, *datasets):
        for dataset in datasets:
            assert (
                issubclass(type(dataset), chainer.dataset.DatasetMixin)), type(
                dataset)
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def get_example(self, i):
        if i < 0:
            raise IndexError
        for dataset in self._datasets:
            if i < len(dataset):
                return dataset[i]
            i -= len(dataset)
        raise IndexError


class SSDMultiboxTrainChain(chainer.Chain):
    def __init__(self, model, alpha=1, k=3):
        super(SSDMultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class SSDTransform(object):
    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))
        img = np.clip(img, a_min=0.0, a_max=255.0)

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)
        return img, mb_loc, mb_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--subset', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--data_type', default='clipart',
                        choices=opt.data_types)
    parser.add_argument('--det_type', choices=['ssd300'],
                        default='ssd300')
    parser.add_argument('--resume',
                        help='path of the model to resume from')
    parser.add_argument('--load', help='load original trained model')
    parser.add_argument('--eval_root')

    # Optional hyper parameters that you can change
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--init_lr', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--log_interval', type=tuple,
                        default=(10, 'iteration'))
    parser.add_argument('--snapshot_interval', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=250)
    args = parser.parse_args()

    datasets_train = get_detection_dataset(args.data_type, args.subset,
                                           args.root)
    dataset_test = get_detection_dataset(args.data_type, 'test',
                                         args.eval_root if args.eval_root else args.root)

    model_args = {'n_fg_class': len(voc_bbox_label_names),
                  'pretrained_model': 'voc0712'}
    model = getattr(chainercv.links, args.det_type.upper())(**model_args)

    if not os.path.exists(args.result):
        os.mkdir(args.result)

    if args.load:
        chainer.serializers.load_npz(args.load, model)

    model.use_preset('evaluate')
    train_chain = SSDMultiboxTrainChain(model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    train = TransformDataset(
        datasets_train,
        SSDTransform(model.coder, model.insize, model.mean))

    train_iter = MultiprocessIterator(train, args.batchsize, n_processes=4,
                                      shared_mem=100000000)
    test_iter = SerialIterator(
        dataset_test, args.batchsize, repeat=False, shuffle=False)

    optimizer = chainer.optimizers.MomentumSGD(lr=args.init_lr)
    optimizer.setup(train_chain)

    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (args.max_iter, 'iteration'),
                               args.result)

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=voc_bbox_label_names),
        trigger=(args.eval_interval, 'iteration'))

    trainer.extend(extensions.LogReport(trigger=args.log_interval))
    trainer.extend(extensions.observe_lr(), trigger=args.log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']),
        trigger=args.log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.snapshot(), trigger=(args.max_iter, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(args.snapshot_interval, 'iteration'))

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'main/loss/loc', 'main/loss/conf'],
                'iteration', trigger=args.log_interval,
                file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['validation/main/map'], 'iteration',
                trigger=(args.eval_interval, 'iteration'),
                file_name='map.png'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()
