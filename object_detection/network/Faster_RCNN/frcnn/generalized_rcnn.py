# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals['boxes'], images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal['losses'])

        if self.training:
            return losses

        return detections

    # def forward(self, images, targets=None):
    #     """
    #     Arguments:
    #         images (list[Tensor]): images to be processed
    #         targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
    #
    #     Returns:
    #         result (list[BoxList] or dict[Tensor]): the output from the model.
    #             During training, it returns a dict[Tensor] which contains the losses.
    #             During testing, it returns list[BoxList] contains additional fields
    #             like `scores`, `labels` and `mask` (for Mask R-CNN models).
    #
    #     """
    #     if self.training and targets is None:
    #         raise ValueError("In training mode, targets should be passed")
    #     original_image_sizes = [img.shape[-2:] for img in images]
    #     images, targets = self.transform(images, targets)
    #     features = self.backbone(images.tensors)
    #     if isinstance(features, torch.Tensor):
    #         features = OrderedDict([(0, features)])
    #     proposals = self.rpn(images, features, targets)
    #     # detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    #     # import pudb
    #     # pudb.set_trace()
    #     # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
    #     # proposals = self.transform.postprocess(proposals, images.image_sizes, original_image_sizes)
    #     boxes = proposals['boxes']  # type:list
    #     scores = proposals['scores']
    #     res = []
    #     # import pudb
    #     # pudb.set_trace()
    #     for i in range(len(proposals['boxes'])):
    #         bs = torch.cat((scores[i].unsqueeze(1), boxes[i]), 1)
    #         bs.sort(0, descending=True)
    #         res.append({'boxes': bs[:100, 1:], 'scores': bs[:100, 0],
    #                     'labels': torch.zeros(100)})
    #     losses = {}
    #     losses.update(proposals['losses'])
    #
    #     if self.training:
    #         return losses
    #
    #     return res
