import math
import torch
import torchvision
import time
from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms
from torchvision.ops import clip_boxes_to_image
from torch.nn.functional import one_hot
from torchvision.ops import box_convert


import torch
from torch import nn

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss
from torch.nn.functional import binary_cross_entropy_with_logits

#Permutation function for Forward pass
def permutTensor(con_out,num_classes):
    N,C,H,W=con_out.shape
    con_out=con_out.view(N,-1,num_classes,H,W)
    con_out=con_out.permute(0,3,4,1,2)
    con_out=con_out.reshape(N,-1,num_classes)
    return con_out

def _fake_cast_onnx(v) -> int:
    return v  # type: ignore[return-value]
def _topk_min(input, orig_kval: int, axis: int) -> int:
    """
    Args:
        input (Tensor): The original input tensor.
        orig_kval (int): The provided k-value.
        axis(int): Axis along which we retrieve the input size.

    Returns:
        min_kval (int): Appropriately selected k-value.
    """
    axis_dim_val = torch._shape_as_tensor(input)[axis].unsqueeze(0)
    min_kval = torch.min(torch.cat((torch.tensor([orig_kval], dtype=axis_dim_val.dtype), axis_dim_val), 0))
    return _fake_cast_onnx(min_kval)
# def nms(boxes, scores, iou_threshold):
#     """
   
#     Args:
#         boxes (Tensor[N, 4])): boxes to perform NMS on. They
#             are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
#             ``0 <= y1 < y2``.
#         scores (Tensor[N]): scores for each one of the boxes
#         iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

#     Returns:
#         Tensor: int64 tensor with the indices of the elements that have been kept
#         by NMS, sorted in decreasing order of scores
#     """
#     return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

# def _batched_nms_vanilla(boxes,scores,idxs,iou_threshold):
#     # Based on Detectron2 implementation, just manually call nms() on each class independently
#     keep_mask = torch.zeros_like(scores, dtype=torch.bool)
#     for class_id in torch.unique(idxs):
#         curr_indices = torch.where(idxs == class_id)[0]
#         curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
#         keep_mask[curr_indices[curr_keep_indices]] = True
#     keep_indices = torch.where(keep_mask)[0]
#     return keep_indices[scores[keep_indices].sort(descending=True)[1]]

# def _batched_nms_coordinate_trick(boxes,scores,idxs,iou_threshold):
#     # strategy: in order to perform NMS independently per class,
#     # we add an offset to all the boxes. The offset is dependent
#     # only on the class idx, and is large enough so that boxes
#     # from different classes do not overlap
#     if boxes.numel() == 0:
#         return torch.empty((0,), dtype=torch.int64, device=boxes.device)
#     max_coordinate = boxes.max()
#     offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
#     boxes_for_nms = boxes + offsets[:, None]
#     keep = nms(boxes_for_nms, scores, iou_threshold)
#     return keep


# def batched_nms(boxes,scores,idxs,iou_threshold):
#     """
#     Performs non-maximum suppression in a batched fashion.

#     Each index value correspond to a category, and NMS
#     will not be applied between elements of different categories.

#     Args:
#         boxes (Tensor[N, 4]): boxes where NMS will be performed. They
#             are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
#             ``0 <= y1 < y2``.
#         scores (Tensor[N]): scores for each one of the boxes
#         idxs (Tensor[N]): indices of the categories for each one of the boxes.
#         iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

#     Returns:
#         Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
#         in decreasing order of scores
#     """
#     if boxes.numel() > (4000 if boxes.device.type == "cpu" else 20000) and not torchvision._is_tracing():
#         return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
#     else:
#         return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)

class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the feature map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        output=[]
        for i in x:
            con_out=self.conv(i)
            con_out=self.cls_logits(con_out)

            if self.training:
                output.append(con_out)
            else:
                #Without permutation the inference results were weird testing if permutation results in better results.
                con_out=permutTensor(con_out,self.num_classes)
                output.append(con_out)
        return output


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        #num_anchors=1
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        regOut=[]
        centOut=[]
        for i in x:
            con_out=self.conv(i)
            #Can try sending bbox_reg thru sigmoid
            reg_out=self.bbox_reg(con_out)
            cent_out=self.bbox_ctrness(con_out)

            if self.training:   
                regOut.append(reg_out)
                centOut.append(cent_out)
            else:
                # Permuting 
                reg_out=permutTensor(reg_out,4)
                cent_out=permutTensor(cent_out,1)
                regOut.append(reg_out)
                centOut.append(cent_out)

        return regOut, centOut


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
        # devices,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function changes depending on if the model is
    in training or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        BIG_NUMBER = 1e8

        positive_samples = 0
        
        cls_loss, reg_loss, ctr_loss = [], [], []
        
        for layer in range(len(cls_logits)):
            cls_logits[layer]   = cls_logits[layer].permute((0,2,3,1))      # N_images x H x W x 20
            reg_outputs[layer]  = reg_outputs[layer].permute((0,2,3,1))     # N_images x H x W x 4
            ctr_logits[layer]   = ctr_logits[layer].permute((0,2,3,1))      # N_images x H x W x 1

        for image, tgt_per_img in enumerate(targets):
            tgt_boxes    = tgt_per_img['boxes']                                                                # N_boxes x 4   (x1,y1,x2,y2)
            tgt_label    = tgt_per_img['labels']                                                               # N_boxes x 1   (class in range [1:20])
            tgt_areas    = (tgt_boxes[:,2] - tgt_boxes[:,0]) * (tgt_boxes[:,3] - tgt_boxes[:,1])         # N_boxes x 1   (area)
            tgt_centers  = (tgt_boxes[:,:2] + tgt_boxes[:,2:]) / 2                                             # N_boxes x 2   (x,y)

            for lyr_points, lyr_stride, lyr_reg_range, lyr_cls_logits, lyr_reg_outputs, lyr_ctr_logits in \
                zip(points, strides, reg_range, cls_logits, reg_outputs, ctr_logits):

                cls_logits_per_point    = lyr_cls_logits[image]       # H x W x 20
                reg_outputs_pp   = lyr_reg_outputs[image]      # H x W x 4
                ctr_logits_per_point    = lyr_ctr_logits[image]       # H x W x 1

                W, H = lyr_points.shape[:2]       # layer_points - H x W x 2      (x,y)
                N_boxes = tgt_boxes.shape[0]     # target_boxes - N_boxes x 4    (x1,y1,x2,y2)

                ctr_box_x1y1 = tgt_centers - self.center_sampling_radius * lyr_stride         # N_boxes x 2
                ctr_box_x2y2 = tgt_centers + self.center_sampling_radius * lyr_stride         # N_boxes x 2
                tgt_ctr_box = torch.concat((ctr_box_x1y1, ctr_box_x2y2), dim = 1)     # N_boxes x 4
                
                repeated_lyr_pts = lyr_points.unsqueeze(dim=0).repeat(N_boxes, 1, 1, 1)          # convert layer_points from H*W*2 to N_boxes*H*W*2
                repeated_tgt_subbox = tgt_ctr_box.view(-1, 1, 1, 4).repeat(1, W, H, 1)     # convert target_center_boxes from N_boxes*4 to N_boxes*H*W*4

                point_x = repeated_lyr_pts[:,:,:,1]            # N_boxes x H x W
                point_y = repeated_lyr_pts[:,:,:,0]            # N_boxes x H x W
                
                subbox_x1 = repeated_tgt_subbox[:,:,:,0]       # N_boxes x H x W
                subbox_y1 = repeated_tgt_subbox[:,:,:,1]       # N_boxes x H x W
                subbox_x2 = repeated_tgt_subbox[:,:,:,2]       # N_boxes x H x W
                subbox_y2 = repeated_tgt_subbox[:,:,:,3]       # N_boxes x H x W
                
                repeated_tgt_box = tgt_boxes.view(-1, 1, 1, 4).repeat(1, W, H, 1)  
                tgt_box_x1 = repeated_tgt_box[:,:,:,0]       # N_boxes x H x W
                tgt_box_y1 = repeated_tgt_box[:,:,:,1]       # N_boxes x H x W
                tgt_box_x2 = repeated_tgt_box[:,:,:,2]       # N_boxes x H x W
                tgt_box_y2 = repeated_tgt_box[:,:,:,3]       # N_boxes x H x W

                l = (point_x - tgt_box_x1)             # N_boxes x H x W
                t = (point_y - tgt_box_y1)             # N_boxes x H x W
                r = (tgt_box_x2 - point_x)             # N_boxes x H x W
                b = (tgt_box_y2 - point_y)             # N_boxes x H x W
                
                # add extra dimension at the end so that concat on last dim returns tensor of shape (N_boxes x H x W x 4)
                #why ??
                ll = l.unsqueeze(-1)                                 # N_boxes x H x W x 1
                tt = t.unsqueeze(-1)                                 # N_boxes x H x W x 1
                rr = r.unsqueeze(-1)                                 # N_boxes x H x W x 1
                bb = b.unsqueeze(-1)                                 # N_boxes x H x W x 1

                max_dist = torch.max(torch.cat((ll, tt, rr, bb), dim=-1), dim=-1)[0]   # N_boxes x H x W

                # each point must satisfy below conditions:
                # 1. point (x,y) must be within target subbox
                # 2. point (x,y) must be within target box      (required as some certain edge-cases lie within target subbox but outside target box)
                # 3. max(l,t,r,b) for that point must be within layer_reg_range

                mask = torch.where(                     # N_boxes x H x W   (True/False)
                    # point lies in target-subbox
                    (point_x >= subbox_x1) & 
                    (point_x <= subbox_x2) &
                    (point_y >= subbox_y1) & 
                    (point_y <= subbox_y2) & 

                    # point lies in target-box
                    (point_x >= tgt_box_x1) & 
                    (point_x <= tgt_box_x2) &
                    (point_y >= tgt_box_y1) & 
                    (point_y <= tgt_box_y2) & 

                    # max distance lies within regression range
                    (max_dist >= lyr_reg_range[0]) & 
                    (max_dist <= lyr_reg_range[1]), 
                    True, False)
                
                foreground_mask = torch.any(mask, dim=0)        # H x W   (True iff point lies inside any box)
                background_mask = ~foreground_mask              # H x W   (True iff point doesn't lie in any box)
                
                # update positive samples
                N_foreground_points = foreground_mask.sum().detach()
                positive_samples += N_foreground_points

                # for each foreground point, find the bounding box area,
                # if point lies in multiple boxes, then multiple values 
                # across N_boxes will be positive. Select the one with
                # least area.
                area_per_point = mask * tgt_areas[:, None, None]             # N_boxes x H x W   (put box-area for foreground points)
                area_per_point[~mask] = BIG_NUMBER                              # N_boxes x H x W   (put BIG_NUMBER for background points)
                box_per_point = torch.min(area_per_point, dim=0)[1]             # H x W             (put box-index [0:N_boxes] of box with least area)
                box_per_point[background_mask] = -1                             # H x W             (put -1 for background and [0:N_boxes] for foreground points)

                label_per_point = tgt_label[box_per_point[foreground_mask]]  # 1 x N_foreground_points

                tgt_class_per_point = box_per_point.unsqueeze(dim=-1).repeat(1, 1, self.num_classes)             # H x W x N_classes
                tgt_class_per_point[background_mask] = 0                                                         # background = zeroes
                tgt_class_per_point[foreground_mask] = one_hot(label_per_point, num_classes=self.num_classes)    # foreground = one-hot
                tgt_class_per_point.detach()


                #####################
                # Classification Loss
                #####################

                # cls_logits_per_point      : (H x W x 20)
                # target_class_per_point    : (H x W x 20)
                cls_loss.append(sigmoid_focal_loss(cls_logits_per_point, tgt_class_per_point, reduction="sum"))

                if N_foreground_points == 0:
                    # skip regression and centerness loss if no foreground points found
                    continue

                #################
                # Regression Loss
                #################
                
                # reg_outputs_per_point     : (H x W x 4)
                predicted_l = reg_outputs_pp[foreground_mask][:,0]       # (N_foreground,)     (predicted distance for each point in foreground)
                predicted_t = reg_outputs_pp[foreground_mask][:,1]
                predicted_r = reg_outputs_pp[foreground_mask][:,2]
                predicted_b = reg_outputs_pp[foreground_mask][:,3]
                
                foreground_x = lyr_points[foreground_mask][:,1]
                foreground_y = lyr_points[foreground_mask][:,0]
                
                predicted_x1 = foreground_x - (predicted_l) * lyr_stride
                predicted_y1 = foreground_y - (predicted_t) * lyr_stride    
                predicted_x2 = foreground_x + (predicted_r) * lyr_stride    
                predicted_y2 = foreground_y + (predicted_b) * lyr_stride 
                predicted_xyxy = torch.stack((predicted_x1, predicted_y1, predicted_x2, predicted_y2), dim=1)       # (N_foreground, 4)
                
                target_xyxy = torch.zeros((*box_per_point.shape, 4), device=predicted_xyxy.device)                        # (H x W x 4)
                target_xyxy[foreground_mask] = tgt_boxes[box_per_point[foreground_mask]]                         # (H x W x 4)
                target_xyxy = target_xyxy[foreground_mask]                                                          # (N_foreground, 4)
                target_xyxy.detach()

                # predicted_xyxy    : (N_foreground, 4)
                # target_xyxy       : (N_foreground, 4)

                reg_loss.append(giou_loss(predicted_xyxy, target_xyxy, reduction="sum"))


                #################
                # Centerness Loss
                #################

                # column vector that specifies the box-index of each foreground point
                box_per_foreground_pt = box_per_point[foreground_mask].view(-1, 1)       # (N_foreground,)
                
                # permute l from (N_boxes x H x W) to (H x W x N_boxes) so that 
                # we can apply foreground mask of same starting dimensions (H x W)
                # Note that in (H x W x N_boxes), each pixel contains l-values for all boxes
                # Then we gather/select l-values of boxes corresponding to box-index specified in the index column vector
                l_foreground = l.permute(1,2,0)[foreground_mask]                            # (N_foreground, N_boxes)
                l_foreground = l_foreground.gather(1, box_per_foreground_pt)             # (N_foreground,)

                # same for other distances
                t_foreground = t.permute(1,2,0)[foreground_mask].gather(1, box_per_foreground_pt)
                r_foreground = r.permute(1,2,0)[foreground_mask].gather(1, box_per_foreground_pt)
                b_foreground = b.permute(1,2,0)[foreground_mask].gather(1, box_per_foreground_pt)
                
                target_centerness =  torch.sqrt(                                                            # (N_foreground,)
                    (torch.min(l_foreground, r_foreground) * torch.min(t_foreground, b_foreground)) / 
                    (torch.max(l_foreground, r_foreground) * torch.max(t_foreground, b_foreground))
                ).detach()

                # ctr_logits_per_point : (H x W x 1)
                predicted_centerness = ctr_logits_per_point[foreground_mask]                    # (N_foreground,1)
                ctr_loss.append(binary_cross_entropy_with_logits(predicted_centerness, target_centerness, reduction="sum"))


        # print(cls_loss, reg_loss, ctr_loss)
        
        positive_samples = max(1, positive_samples)

        # torch.stack(cls_loss) - N x H x W
        total_cls_loss = torch.sum(sum(cls_loss)) / positive_samples   # scalar
        total_reg_loss = torch.sum(sum(reg_loss)) / positive_samples   # scalar
        total_ctr_loss = torch.sum(sum(ctr_loss)) / positive_samples   # scalar
        final_loss = total_cls_loss + total_reg_loss + total_ctr_loss

        return {
            'cls_loss'  : total_cls_loss,
            'reg_loss'  : total_reg_loss,
            'ctr_loss'  : total_ctr_loss,
            'final_loss': final_loss
        }


    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """
    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        
        detections = []
        num_of_images = len(image_shapes)
        #TODO Can image be matricied
        pointscat=[]
        for i in range(len(points)):
            pointscat.append(points[i].reshape(-1,2))
        for index in range(num_of_images):
            box_regression_per_image = [br[index] for br in reg_outputs]
            logits_per_image = [cl[index] for cl in cls_logits]
            box_ctrness_per_image = [bc[index] for bc in ctr_logits]
            image_boxes = []
            image_scores = []
            image_labels = []
            i=0
            for box_regression_per_level, logits_per_level, box_ctrness_per_level in zip(box_regression_per_image, logits_per_image, box_ctrness_per_image):
                num_classes = logits_per_level.shape[-1]
                #print(i)
                # remove low scoring boxes
                scores_per_level = torch.sqrt(
                    torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)
                ).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = _topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]
                box_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes +1

                pred_l, pred_t, pred_r, pred_b = box_regression_per_level.permute(1,0)
                centerx = pointscat[i][:,1]
                centery = pointscat[i][:,0]
                # x1 = math.floor(strides[i]/2)+(centerx - pred_l * strides[i])
                # y1 = math.floor(strides[i]/2)+(centery - pred_b * strides[i])
                # x2 = math.floor(strides[i]/2)+(centerx + pred_r * strides[i])
                # y2 = math.floor(strides[i]/2)+(centery + pred_t * strides[i])

                x1 = centerx-(math.floor(strides[i]/2) + pred_l * strides[i])
                y1 = centery-(math.floor(strides[i]/2) + pred_b * strides[i])
                x2 = centerx+(math.floor(strides[i]/2) + pred_r * strides[i])
                y2 = centery+(math.floor(strides[i]/2) + pred_t * strides[i])
                
                boxes_per_level=torch.stack([x1,y1,x2,y2]).permute(1,0)
                boxes_per_level = clip_boxes_to_image(boxes_per_level, image_shapes[index])


                image_boxes.append(boxes_per_level[box_idxs])
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)
                i+=1
            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections
