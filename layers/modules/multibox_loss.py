# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(
        self,
        num_classes,
        overlap_thresh,
        prior_for_matching,
        bkg_label,
        neg_mining,
        neg_pos,
        neg_overlap,
        encode_target,
        use_gpu=True,
    ):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg["variance"]

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0) # num here=batch size
        priors = priors[: loc_data.size(1), :] # priors = priors[:8732, :]
        num_priors = priors.size(0) # num_priors=8732
        num_classes = self.num_classes # num_classes = 3

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)# 定义目标定位值的tensor维度,size:(batch size, 8732, 4)
        conf_t = torch.LongTensor(num, num_priors)# 定义目标置信度的tensor维度,size:(batch size, 8732)
        for idx in range(num):
            truths = targets[idx][:, :-1].data # targets[0]中第一个到倒数第二个数值
            labels = targets[idx][:, -1].data # targets[0]中最后一个数值为label
            defaults = priors.data
            #在训练时，groundtruth boxes 与 default boxes（就是prior boxes） 按照如下方式进行配对：
            # 首先，寻找与每一个ground truth box有最大的jaccard overlap的default box，这样就能保证每一个groundtruth box与唯一的一个default box对应起来（所谓的jaccard overlap就是IoU）。
            # SSD之后又将剩余还没有配对的default box与任意一个groundtruth box尝试配对，只要两者之间的jaccard overlap大于阈值，就认为match（SSD 300 阈值为0.5）。
            # 显然配对到GT的default box就是positive，没有配对到GT的default box就是negative。
            match(
                self.threshold,
                truths,
                defaults,
                self.variance,
                labels,
                loc_t,
                conf_t,
                idx,
            )
        if self.use_gpu: # cpu到gpu的数据转换
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False) 
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0 # pos.shape: torch.Size([batch size, 8732]),conf_t中所有大于0的地方，pos在该位置为1
        num_pos = pos.sum(dim=1, keepdim=True) # num_pos.shape: torch.Size([batch size, 1]) #每张图可匹配默认框数量

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data) # 增一维度size扩展为(32,8732,1) 再扩增为loc_data(32,8732,4)的维度
        loc_p = loc_data[pos_idx].view(-1, 4) # 此时loc_data和pos_idx维度一样，选择出positive的loc
        loc_t = loc_t[pos_idx].view(-1, 4) # 选出一样数量的目标定位值
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) # 用smooth_l1_los求损失

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1) # 对于每个GT, idx_rank是该位置的priors的loss得分排名
        num_pos = pos.long().sum(1, keepdim=True)
        #clamp:将输入input张量每个元素的夹紧到区间 [min,max]
        #由于负样本数目远大于正样本数，因此将正负比例控制在1：3 -> negpos_ratio=3
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        # 从所有idx_rank中挑选出在负样本范围内的样本id
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            
        # conf_data shape [batch,num_priors,num_classes]
        for idx in range(num):
            labels = targets[idx][:, -1].data
            if labels[0] == 0:
                conf_data[idx][:][1] *= 5
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().double() # N为batch size个图像中检测到的positive框总数
        loss_l = loss_l.double() / N
        loss_c = loss_c.double() / N
        return loss_l, loss_c
