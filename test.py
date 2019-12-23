from __future__ import print_function

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from SIXray import SIXray_CLASSES as labelmap, SIXrayDetection, BaseTransform, SIXrayAnnotationTransform
from ssd import build_ssd
from config import HOME


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_XRAY_10000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--xray_root', default='', help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
parser.add_argument('--image_folder', default=os.path.abspath('data_sets/core_3000/Image'), type=str)
parser.add_argument('--annotation_folder', default=os.path.abspath('data_sets/core_3000/Annotation'), type=str)

args = parser.parse_args()
dataset_mean = (104, 117, 123)
set_type = 'test'

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def test(img_folder, anno_folder):
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = SIXrayDetection(
        img_folder, anno_folder,  BaseTransform(net.size, (104, 117, 123)), SIXrayAnnotationTransform()
    )

    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]


    for i in range(num_images):
        print("Testing image {:d}/{:d}....".format(i + 1, num_images))
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        detections = net(x).data
        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

    for cls_ind, cls in enumerate(labelmap):
        # print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects a1-based indices
                for k in range(dets.shape[0]):
                    if dets[k, -1] > 0.01:
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

    return 1


def get_voc_results_file_template(image_set, cls):
    filename = 'det_' + image_set + '_{}.txt'.format(cls)
    filedir = 'eval'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


if __name__ == '__main__':
    anno_folder = 'c:/Users/orang/work/Anno_test'
    img_folder = 'c:/Users/orang/work/Image_test'
    test(img_folder, anno_folder)