import argparse

import chainer
import matplotlib.pyplot as plt
from chainercv import utils
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.visualizations import vis_bbox

chainer.config.train = False
parser = argparse.ArgumentParser()
parser.add_argument('image', help='path for input image')
parser.add_argument('result', help='path for output image')
parser.add_argument('--load', help='if not specified, use default model \
                                                trained on voc07+12 trainval')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--score_thresh', type=float, default=0.25)
args = parser.parse_args()

model = SSD300(
    n_fg_class=len(voc_bbox_label_names), pretrained_model='voc0712')
model.score_thresh = args.score_thresh

if args.load:
    chainer.serializers.load_npz(args.load, model)

if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

img = utils.read_image(args.image, color=True)
bboxes, labels, scores = model.predict([img])
vis_bbox(
    img, bboxes[0], labels[0], scores[0], label_names=voc_bbox_label_names)
plt.axis('off')
plt.tight_layout()
plt.savefig(args.result)
