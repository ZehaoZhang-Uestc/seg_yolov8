import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize

class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None,num_classes=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        # self.criterion = nn.NLLLoss(ignore_index=15)
        weights = torch.full((16,), 10.0)
        # weights = torch.full((2,), 10.0)
        weights[-1] = 1
        self.criterion = nn.NLLLoss()

    def forward(self, predict, target):
        target = target.long()
        return self.criterion(self.softmax(predict), target)


class MaskANN(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels,
                 feat_channels=256,
                 stacked_convs=3,
                 kernel_size=(3, 3, 3),
                 dilation=(2, 2, 2),
                 padding=(2, 2, 2),
                 class_channel=16
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.enrich_semantics_supervised_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.out_channels
            self.enrich_semantics_supervised_convs.append(
                nn.Conv2d(in_channels=chn, out_channels=self.out_channels, kernel_size=self.kernel_size[i],
                          stride=1, dilation=self.dilation[i], padding=self.padding[i]))
        self.enrich_semantics_supervised_convs.append(nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1))

        self.mask_layer = nn.Conv2d(self.out_channels, out_channels=self.num_classes+1, kernel_size=1, stride=1, )

        self.dot_layer = nn.Sequential(nn.Conv2d(self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1, ),
                                       nn.Sigmoid())


    def forward(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        mask_feat = x

        for mask_conv in self.enrich_semantics_supervised_convs:
            mask_feat = mask_conv(mask_feat)

        mask = self.mask_layer(mask_feat)
        dotmap = self.dot_layer(mask_feat)
        OUT_feat = dotmap * x
        return mask, OUT_feat


class MaskDenoiseHead(nn.Module):
    def __init__(self,
                 num_classes=15,
                 in_channels=(256, 512, 512), #n: (256, 128, 256)
                 stage=3,
                 feat_channels=256,
                 out_channels=(256, 512, 512),
                 stacked_convs=3,
                 kernel_size=(3, 3, 3),
                 dilation=(2, 2, 2),
                 padding=(2, 2, 2),
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stage = stage
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self._init_blocks()

    def _init_blocks(self):

        for i in range(self.stage):
            Mask_block = MaskANN(num_classes= self.num_classes, in_channels=self.in_channels[i],
                                 out_channels=self.out_channels[i], feat_channels=self.feat_channels, stacked_convs=self.stacked_convs,
                                 kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding)
            setattr(self, f'Denoise_Block{i + 1}', Mask_block)

    def forward(self, x):
        Out_Pre = []
        Pred_Mask = []
        for i in range(self.stage):
            Mask_block = getattr(self, f'Denoise_Block{i + 1}')
            pred_mask, out_pre = Mask_block(x[i])
            Out_Pre.append(out_pre)
            Pred_Mask.append(pred_mask)
        return tuple(Pred_Mask), Out_Pre


    def forward_train(self,
                      mask_pred,
                      mask_img,
                      # gt_masks = None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = mask_pred
        mask_loss = []
        for out in outs:
            (b, c, h, w) = out.size()
            True_mask = Resize([h, w], interpolation=0)(mask_img)
            losses = self.loss(out, True_mask)
            mask_loss.append(losses)
        return mask_loss

