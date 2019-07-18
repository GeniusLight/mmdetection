from .two_stage import TwoStageDetector
from ..registry import DETECTORS
from ..losses import CtdetLoss
import torch


@DETECTORS.register_module
class CenterNet(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(CenterNet, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.loss = CtdetLoss()

    def forward_train(self, img, img_meta, **kwargs):
        # print('in forward train')
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        # print(kwargs)
        # loss, loss_stats = self.loss(output, **kwargs)
        losses = self.loss(output, **kwargs)

        # import pdb; pdb.set_trace()
        return losses#, loss_stats
