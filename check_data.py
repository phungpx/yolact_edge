import numpy as np

from yolact_edge.data import *
from yolact_edge.utils.augmentations import SSDAugmentation
import torch
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--config', default='yolact_edge_mobilenetv2_config',
                    help='The config object to use.')
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


dataset = COCODetection(image_path=cfg.dataset.train_images,
                                info_file=cfg.dataset.train_info,
                                transform=SSDAugmentation(MEANS))


for i in range(50):
    im, (gt, masks, num_crowds) = dataset.__getitem__(i)
    img = np.transpose(im.numpy(), (1, 2, 0)) * 255
    img = np.clip(img, a_min=0, a_max=255)
    masks = np.transpose(masks, (1, 2, 0)) * 255
    name = str(i) + '.jpg'
    m_name = str(i) + '_.jpg'
    cv2.imwrite(name, img)
    cv2.imwrite(m_name, masks)
    a = 0
