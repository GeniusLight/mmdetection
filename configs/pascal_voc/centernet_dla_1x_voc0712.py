# model settings
model = dict(
    type='CenterNet',
    pretrained='modelzoo://centernet_hg',
    backbone=dict(
        type='DLA',
        base_name='dla34',
        last_level=6),
    rpn_head=dict(
        type='CtdetHead', heads=dict(hm=20, wh=2, reg=2)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=64,
        featmap_strides=[4]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=64,
        fc_out_channels=256,
        roi_feat_size=7,
        num_classes=20,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
cudnn_benchmark = True

train_cfg = dict(
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.1,
            neg_iou_thr=0.1,
            min_pos_iou=0.1,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))

_valid_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
]
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

test_cfg = dict(
    num_classes=20,
    valid_ids={i + 1: v
               for i, v in enumerate(_valid_ids)},
    img_norm_cfg=img_norm_cfg,
    debug=0,
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))

import numpy as np
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CtdetTrainTransforms',
            flip_ratio=0.5,
            size_divisor=31,
            keep_ratio=False,
            img_scale=(384,384),
            img_norm_cfg=img_norm_cfg,
            _data_rng = np.random.RandomState(123),
            _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32),
            _eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
                                  [-0.5832747, 0.00994535, -0.81221408],
                                  [-0.56089297, 0.71832671, 0.41158938]],
                                 dtype=np.float32),
            max_objs = 50,
            num_classes = 20)
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1, 1),
        flip=False,
        transforms=[
            dict(type='CtdetTestTransforms',
                size_divisor=31,
                keep_ratio=False,
                input_res=(384, 384),
                img_norm_cfg=img_norm_cfg)
        ])
]

dataset_type = 'VOCDataset'
data_root = 'data/voc/'
# data = dict(
#     imgs_per_gpu=3,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         ann_file=[
#             data_root + 'VOC2007/ImageSets/Main/trainval.txt',
#             data_root + 'VOC2012/ImageSets/Main/trainval.txt'
#         ],
#         img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
#         img_prefix=data_root + 'VOC2007/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
#         img_prefix=data_root + 'VOC2007/',
#         pipeline=test_pipeline))
data = dict(
    imgs_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[3])  # actual epoch = 3 * 3 = 9

# # optimizer
# optimizer = dict(type='Adam', lr=1.25e-4)
# # optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = {}
# # learning policy
# lr_config = dict(
#     policy='step',
#     # warmup='linear',
#     # warmup_iters=500,
#     # warmup_ratio=1.0 / 3,
#     step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 4
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'data/work_dirs/centernet_dla_rcnn_pascal'
load_from = 'data/work_dirs/centernet_dla_pascal_normal70_1/epoch_70.pth'
resume_from = None
workflow = [('train', 1)]
