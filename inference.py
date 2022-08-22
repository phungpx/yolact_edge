#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import argparse
import cv2
import torch.backends.cudnn as cudnn
import numpy as np

from flame.core.model.yolact import Yolact
from configs.config import get_config
from flame.utils import timer
from flame.utils.output_utils import nms, after_nms, draw_img

parser = argparse.ArgumentParser(description='YOLACT Detection.')
parser.add_argument('--cfg', default='res50_custom', help='The configuration name to use.')
parser.add_argument('--weight', default='weights/model_best.pth', type=str)
parser.add_argument('--image', default=None, type=str, help='The folder of images for detecting.')
parser.add_argument('--video', default=None, type=str, help='The path of the video to evaluate.')
parser.add_argument('--img_size', type=int, default=550, help='The image size for validation.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--hide_mask', default=False, action='store_true', help='Hide masks in results.')
parser.add_argument('--hide_bbox', default=False, action='store_true', help='Hide boxes in results.')
parser.add_argument('--hide_score', default=False, action='store_true', help='Hide scores in results.')
parser.add_argument('--cutout', default=False, action='store_true', help='Cut out each object and save.')
parser.add_argument('--save_lincomb', default=False, action='store_true', help='Show the generating process of masks.')
parser.add_argument('--no_crop', default=False, action='store_true',
                    help='Do not crop the output masks with the predicted bounding box.')
parser.add_argument('--real_time', default=False, action='store_true', help='Show the detection results real-timely.')
parser.add_argument('--visual_thre', default=0.3, type=float,
                    help='Detections with a score under this threshold will be removed.')

args = parser.parse_args()
# args.cfg = re.findall(r'res.+_[a-z]+', args.weight)[0]
cfg = get_config(args, mode='detect')

net = Yolact(cfg)
net.load_weights(cfg.weight, cfg.cuda)
net.eval()
print(f'Model loaded with {cfg.weight}.\n')

if cfg.cuda:
    cudnn.benchmark = True
    cudnn.fastest = True
    net = net.cuda()

with torch.no_grad():
    # detect images
    img_origin = cv2.imread("/content/PennFudanPed/PNGImages/FudanPed00005.png")
    # img_h, img_w = img_origin.shape[0:2]

    img = cv2.resize(pad_to_square(img_origin), dsize=(550, 550))

    img_origin = img.astype(np.uint8)
    img_h, img_w = img_origin.shape[0:2]


    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.float().div(255.)
    img = torch.unsqueeze(img, 0)

    if cfg.cuda:
        img = img.cuda()

    with timer.counter('forward'):
        class_p, box_p, coef_p, proto_p, anchors = net(img)

    with timer.counter('nms'):
        ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, anchors, cfg)

    print(ids_p, class_p, box_p)
    img_name = 'out.jpg'

    with timer.counter('after_nms'):
        ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p,
                                                     proto_p, img_h, img_w, cfg, img_name=img_name)

    with timer.counter('save_img'):
        img_numpy = draw_img(ids_p, class_p, boxes_p, masks_p, img_origin, cfg, img_name=img_name)
        cv2.imwrite(f'{img_name}', img_numpy)

    print('\nFinished, saved in: results/images.')
