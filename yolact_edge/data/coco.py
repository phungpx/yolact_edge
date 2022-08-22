import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from .config import cfg
from pycocotools import mask as maskUtils
import random
import imgaug.augmenters as iaa

def get_label_map():
    if cfg.dataset.label_map is None:
        return {x+1: x+1 for x in range(len(cfg.dataset.class_names))}
    else:
        return cfg.dataset.label_map

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map()
        print(self.label_map)

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, transform=None,
                 target_transform=COCOAnnotationTransform(),
                 dataset_name='MS COCO', has_gt=True, copy_paste=False):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO

        self.root = image_path
        self.coco = COCO(info_file)

        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())

        self.transform = transform
        self.target_transform = target_transform

        self.name = dataset_name
        self.has_gt = has_gt
        self.copy_paste = copy_paste # whether to copy and paste image to image

        self.filter_dataset_map()
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')

    def filter_dataset_map(self):
        if (cfg.joint_dataset is None or cfg.joint_dataset.dataset_map is None) and cfg.dataset.dataset_map is None:
            return

        from .config import COCO_CLASSES, COCO_LABEL_MAP, COCO_YTVIS_CLASS_MAP, COCO_YTVIS_LABEL_MAP, COCO_INTER_LABEL_MAP

        filtered_ids = []
        ids = []

        self.target_transform.label_map = COCO_INTER_LABEL_MAP if cfg.dataset.dataset_map == 'coco-inter' else COCO_YTVIS_LABEL_MAP

        for index in range(len(self.ids)):
            img_id = self.ids[index]

            if self.has_gt:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)

                # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
                targets = self.coco.loadAnns(ann_ids)
                found_filtered_classes = False
                for target in targets:
                    coco_class_name = COCO_CLASSES[COCO_LABEL_MAP[target['category_id']] - 1]
                    if coco_class_name in COCO_YTVIS_CLASS_MAP:
                        found_filtered_classes = True
                        break
                if found_filtered_classes:
                    ids.append(img_id)
                else:
                    filtered_ids.append(img_id)

        self.ids = ids


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            target = self.coco.imgToAnns[img_id]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
        else:
            target = []

        if (cfg.joint_dataset is not None and cfg.joint_dataset.dataset_map is not None) \
                or cfg.dataset.dataset_map is not None:
            from .config import COCO_CLASSES, COCO_LABEL_MAP, COCO_YTVIS_CLASS_MAP, COCO_YTVIS_LABEL_MAP
            target = [x for x in target
                      if COCO_CLASSES[COCO_LABEL_MAP[x['category_id']] - 1] in COCO_YTVIS_CLASS_MAP]

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path)
        height, width, _ = img.shape

        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = []
            for obj in target:
                mask = self.coco.annToMask(obj)
                if mask.shape == (height, width):
                    masks.append(mask)
                else:
                    print(file_name)
                    masks.append(np.zeros(shape=(height, width)))

            masks = np.stack(masks, axis=0)
            # masks = masks.reshape(-1, height, width)

        # Apply only for DKX dataset
        # Padding images to square
        img = self.pad_to_square(image=img)
        masks = [self.pad_to_square(image=masks[i, :, :]) for i in range(masks.shape[0])]
        masks = np.stack(masks, axis=0)
        height, width, _ = img.shape

        percent = target[0]['area'] / (height * width)
        areas = [target[i]['area'] for i in range(len(target))]

        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        # random copy and paste
        # only apply to images whose card area is less than 20% of padded image areas
        if random.random() <= 0.5 and self.copy_paste and percent <= 0.25:
            # get a random sample from dataset
            copy_id = random.randint(0, self.__len__() - 1)
            c_img, c_masks, c_cate, c_percent, area = self.pull_raw_item(copy_id)

            if c_percent > percent:
                a = 1
                b = (c_img.shape[0] + c_img.shape[1])
                c = (c_img.shape[0] * c_img.shape[1]) - area/percent
                coeff = [a, b, c]
                roots = np.roots(coeff)
                border = int(np.max(roots) / 2)

                if border > 0:
                    c_img = cv2.copyMakeBorder(c_img.copy(), border, border, border, border, cv2.BORDER_CONSTANT, value=[0,0,0])
                    c_masks = cv2.copyMakeBorder(c_masks.copy(), border, border, border, border, cv2.BORDER_CONSTANT, value=[0])

            # Pastes objects and creates a new composed image.
            c_img = cv2.resize(c_img, img.shape[:2])
            c_mask = cv2.resize(c_masks, img.shape[:2])
            compose_mask = c_mask.copy().astype(img.dtype) * 255

            # blur mask
            compose_mask = cv2.GaussianBlur(compose_mask, (9, 9), cv2.BORDER_DEFAULT) / 255.0
            compose_mask = [compose_mask, compose_mask, compose_mask]
            compose_mask = np.stack(compose_mask, axis=2)

            img = img * (1 - compose_mask) + c_img * compose_mask

            masks = [np.where(mask == c_mask, 0, mask) for mask in masks]

            masks.append(c_mask)

            # process
            labels = [target[i][4] for i in range(len(target))]
            labels.append(c_cate - 1)

            for i in range(len(masks) - 1):
                ori_area = areas[i]
                num_class, label = cv2.connectedComponents(masks[i])

                if num_class > 2:
                    masks.pop(i)
                    labels.pop(i)
                else:
                    new_area = (masks[i] == 1).sum()
                    if new_area < ori_area * 0.8:
                        masks.pop(i)
                        labels.pop(i)

            masks = np.stack(masks, axis=0)

            target = self.get_boxes(masks, labels, img.shape[1], img.shape[0])

            # name = str(index) + '.jpg'
            # cv2.imwrite(name, img)
            # for i in range(len(masks)):
            #     mask = np.expand_dims(masks[i], 2) * 255
            #     m_name = str(index) + '_' + str(i) + '_.jpg'
            #     cv2.imwrite(m_name, mask)

        # masks: ndarray (n, h, w)
        # img: ndarray (h, w, 3)
        # target: list([x_min, y_min, x_max, y_max, label])

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                    {'num_crowds': num_crowds, 'labels': target[:, 4]})

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels     = labels['labels']

                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]),
                    {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        if target is not None and target.shape[0] == 0:
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids)-1))

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds


    def pull_raw_item(self, index):
        """
        This function is used only for DKX dataset :))
        """
        img_id = self.ids[index]

        if self.has_gt:
            target = self.coco.imgToAnns[img_id]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
        else:
            target = []

        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path)
        height, width, _ = img.shape

        if len(target) > 0:
            # only get first object in this image
            masks = self.coco.annToMask(target[0]).reshape(-1)
            masks = masks.reshape(height, width)

        # Padding images to square
        img = self.pad_to_square(image=img)
        masks = self.pad_to_square(image=masks)

        # masks: ndarray (n, h, w)
        # img: ndarray (h, w, 3)

        category_id = target[0]['category_id']

        percent = target[0]['area'] / (height * width)


        return img, masks, category_id, percent, target[0]['area']

    def get_boxes(self, masks, labels, w, h):
        '''
        masks: ndarray (n, h, w)
        labels: list()
        '''
        # get boxes in masks
        boxes = []
        for i in range(masks.shape[0]):
            mask = masks[i, :, :]
            pos = np.where(mask == 1)
            xmin, xmax = np.min(pos[1]) / w, np.max(pos[1]) / w
            ymin, ymax = np.min(pos[0]) / h, np.max(pos[0]) / h
            boxes.append([xmin, ymin, xmax, ymax, labels[i]])

        return boxes

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
