import os.path as osp

import mmcv
import numpy as np
import cv2
import math
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .registry import DATASETS
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import (to_tensor, random_scale, color_aug,
                    gaussian_radius, draw_umich_gaussian,
                    get_affine_transform, affine_transform,
                    get_border)
from .extra_aug import ExtraAugmentation


@DATASETS.register_module
class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 with_ctdet=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(ann_file)
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # with heatmap etc needed for ctdet method
        self.with_ctdet = with_ctdet
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        if self.with_ctdet:
            height, width = img.shape[0], img.shape[1]
            img_shape = img.shape
            c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
            if self.resize_keep_ratio:
                input_h = (height | self.size_divisor) + 1
                input_w = (width | self.size_divisor) + 1
                s = np.array([input_w, input_h], dtype=np.float32)
            else:
                s = max(img.shape[0], img.shape[1]) * 1.0
                input_h, input_w = self.img_scales[0]

            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = get_border(128, img.shape[1])
            h_border = get_border(128, img.shape[0])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

            if flip:
                img = img[:, ::-1, :]
                c[0] =  width - c[0] - 1

            trans_input = get_affine_transform(
              c, s, 0, [input_w, input_h])
            inp = cv2.warpAffine(img, trans_input,
                                 (input_w, input_h),
                                 flags=cv2.INTER_LINEAR)

            pad_shape = inp.shape
            scale_factor = np.array(
                [(pad_shape[1]/img_shape[1]), (pad_shape[0]/img_shape[0]),
                (pad_shape[1]/img_shape[1]), (pad_shape[0]/img_shape[0])], dtype=np.float32)

            inp = (inp.astype(np.float32) / 255.)
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
            mean = np.array(self.img_norm_cfg['mean'], dtype=np.float32).reshape(1,1,3)
            std = np.array(self.img_norm_cfg['std'], dtype=np.float32).reshape(1,1,3)
            inp = (inp - mean) / std
            inp = inp.transpose(2, 0, 1)
            img = inp.copy()
        else:
            img_scale = random_scale(self.img_scales, self.multiscale_mode)
            img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
            img = img.copy()

        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals

        if not self.with_ctdet:
            gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                            flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        if self.with_ctdet:
            # inp = (inp.astype(np.float32) / 255.)
            # color_aug(self._data_rng, img, self._eig_val, self._eig_vec)
            # inp = (inp - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
            # inp = inp.transpose(2, 0, 1)

            # TODO: change to down_ratio
            output_h = input_h // 4
            output_w = input_w // 4
            trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

            hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)
            ind = np.zeros((self.max_objs), dtype=np.int64)
            reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

            for k in range(min(len(ann['labels']), self.max_objs)):
                bbox = ann['bboxes'][k]
                cls_id = ann['labels'][k] - 1
                if flip:
                    bbox[[0, 2]] = width - bbox[[2, 0]] - 1

                # tranform bounding box to output size
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    # populate hm based on gd and ct
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array(
                      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_umich_gaussian(hm[cls_id], ct_int, radius)
                    wh[k] = 1. * w, 1. * h
                    ind[k] = ct_int[1] * output_w + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        if self.with_ctdet:
            data['hm'] = DC(to_tensor(hm), stack=True)
            data['reg_mask'] = DC(to_tensor(reg_mask).unsqueeze(1), stack=True, pad_dims=1)
            data['ind'] = DC(to_tensor(ind).unsqueeze(1), stack=True, pad_dims=1)
            data['wh'] = DC(to_tensor(wh), stack=True, pad_dims=1)
            data['reg'] = DC(to_tensor(reg), stack=True, pad_dims=1)

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            if self.with_ctdet:
                height, width = img.shape[0:2]
                new_height = int(height * scale[0])
                new_width  = int(width * scale[1])
                img_shape = (new_height, new_width)
                # if self.opt.fix_res:
                #     inp_height, inp_width = self.opt.input_h, self.opt.input_w
                #     c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
                #     s = max(height, width) * 1.0
                # else:
                inp_height = (new_height | self.size_divisor) + 1
                inp_width = (new_width | self.size_divisor) + 1
                c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
                s = np.array([inp_width, inp_height], dtype=np.float32)

                trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
                resized_image = cv2.resize(img, (new_width, new_height))
                inp_image = cv2.warpAffine(
                    resized_image, trans_input, (inp_width, inp_height),
                    flags=cv2.INTER_LINEAR)
                # img meta calculations
                pad_shape = inp_image.shape[:2]
                scale_factor = np.array(
                    [(pad_shape[1]/img_shape[1]), (pad_shape[0]/img_shape[0]),
                    (pad_shape[1]/img_shape[1]), (pad_shape[0]/img_shape[0])], dtype=np.float32)
                mean = np.array(self.img_norm_cfg['mean'], dtype=np.float32).reshape(1,1,3)
                std = np.array(self.img_norm_cfg['std'], dtype=np.float32).reshape(1,1,3)
                inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

                _img = inp_image.transpose(2, 0, 1)
                if flip:
                    _img = _img[:, :, ::-1].copy()

                _img_meta = dict(
                    ori_shape=(img_info['height'], img_info['width'], 3),
                    img_shape=img_shape,
                    pad_shape=pad_shape,
                    scale_factor=scale_factor,
                    ctdet_c=c,
                    ctdet_s=s,
                    ctdet_out_height=inp_height // 4,
                    ctdet_out_width=inp_width // 4,
                    flip=flip)
                # images = torch.from_numpy(images)
                # meta = {'c': c, 's': s,
                #         'out_height': inp_height // 4,
                #         'out_width': inp_width // 4,
                #         'img_id':img_id,
                #         'mean': self.img_norm_cfg['mean'],
                #         'std': self.img_norm_cfg['std']}
            else:
                _img, img_shape, pad_shape, scale_factor = self.img_transform(
                    img, scale, flip, keep_ratio=self.resize_keep_ratio)
                _img = to_tensor(_img)
                _img_meta = dict(
                    ori_shape=(img_info['height'], img_info['width'], 3),
                    img_shape=img_shape,
                    pad_shape=pad_shape,
                    scale_factor=scale_factor,
                    flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
