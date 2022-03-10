# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from net.backbone import build_backbone
from net.operators import ASPP
from utils.registry import NETS
class _deeplabv2(nn.Module):	
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d):
		super(_deeplabv2, self).__init__()
		self.batchnorm = batchnorm
		self.backbone = build_backbone(self.cfg.MODEL_BACKBONE, os=self.cfg.MODEL_OUTPUT_STRIDE)
		input_channel = self.backbone.OUTPUT_DIM	
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=[6,12,18,24],
				bn_mom = cfg.TRAIN_BN_MOM,
				has_global=cfg.MODEL_ASPP_HASGLOBAL,
				batchnorm=batchnorm
		)
		self.cfg = cfg
	def __initial__(self):
		for m in self.modules():
			if m not in self.backbone.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				elif isinstance(m, self.batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		
	def forward(self, x):
		raise NotImplementedError

@NETS.register_module
class deeplabv2(_deeplabv2):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv2, self).__init__(cfg, batchnorm)
		self.dropout1 = nn.Dropout(0.5)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.__initial__()
		self.from_scratch_layers = [self.cls_conv]
		for m in self.aspp.modules():
			if isinstance(m, nn.Conv2d):
				self.from_scratch_layers.append(m)

	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		feature = self.aspp(x_bottom)
		feature = self.dropout1(feature)
		result = self.cls_conv(feature)
		result = F.interpolate(result,(h,w),mode='bilinear', align_corners=True)

		return result

	def get_parameter_groups(self):
		groups = ([], [], [], [])
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.weight.requires_grad:
					if m in self.from_scratch_layers:
						groups[2].append(m.weight)
						print(m)
					else:
						groups[0].append(m.weight)

				if m.bias is not None and m.bias.requires_grad:
					if m in self.from_scratch_layers:
						groups[3].append(m.bias)
					else:
						groups[1].append(m.bias)
		return groups

