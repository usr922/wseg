# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from net.backbone import build_backbone
from utils.registry import NETS

@NETS.register_module
class deeplabv1(nn.Module):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv1, self).__init__()
		self.cfg = cfg
		self.batchnorm = batchnorm
		#self.backbone = build_backbone(self.cfg.MODEL_BACKBONE, os=self.cfg.MODEL_OUTPUT_STRIDE)
		self.backbone = build_backbone(self.cfg.MODEL_BACKBONE, pretrained=cfg.MODEL_BACKBONE_PRETRAIN, norm_layer=self.batchnorm, **kwargs)
		self.conv_fov = nn.Conv2d(self.backbone.OUTPUT_DIM, 512, 3, 1, padding=12, dilation=12, bias=False)
		self.bn_fov = batchnorm(512, momentum=cfg.TRAIN_BN_MOM, affine=True)
		self.conv_fov2 = nn.Conv2d(512, 512, 1, 1, padding=0, bias=False)
		self.bn_fov2 = batchnorm(512, momentum=cfg.TRAIN_BN_MOM, affine=True)
		self.dropout1 = nn.Dropout(0.5)
		self.cls_conv = nn.Conv2d(512, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.__initial__()
		self.not_training = []#[self.backbone.conv1a, self.backbone.b2, self.backbone.b2_1, self.backbone.b2_2]
		#self.from_scratch_layers = [self.cls_conv]
		self.from_scratch_layers = [self.conv_fov, self.conv_fov2, self.cls_conv]
	
	def __initial__(self):
		for m in self.modules():
			if m not in self.backbone.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				elif isinstance(m, self.batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		#self.backbone = build_backbone(self.cfg.MODEL_BACKBONE, pretrained=self.cfg.MODEL_BACKBONE_PRETRAIN)

	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		feature = self.conv_fov(x_bottom)
		feature = self.bn_fov(feature)
		feature = F.relu(feature, inplace=True)
		feature = self.conv_fov2(feature)
		feature = self.bn_fov2(feature)
		feature = F.relu(feature, inplace=True)
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
					else:
						groups[0].append(m.weight)

				if m.bias is not None and m.bias.requires_grad:

					if m in self.from_scratch_layers:
						groups[3].append(m.bias)
					else:
						groups[1].append(m.bias)
		return groups

@NETS.register_module
class deeplabv1_caffe(nn.Module):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv1_caffe, self).__init__()
		self.cfg = cfg
		self.batchnorm = batchnorm
		self.backbone = build_backbone(self.cfg.MODEL_BACKBONE, os=self.cfg.MODEL_OUTPUT_STRIDE)
		outdim = 4096
		self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
		#self.avgpool = nn.AvgPool2d(3, stride=1, padding=1)
		self.conv_fov = nn.Conv2d(self.backbone.OUTPUT_DIM, outdim, 3, 1, padding=12, dilation=12)
		self.dropout1 = nn.Dropout(0.5)
		self.conv_fov2 = nn.Conv2d(outdim, outdim, 1, 1, padding=0)
		self.dropout2 = nn.Dropout(0.5)
		self.cls_conv = nn.Conv2d(outdim, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.__initial__()
		self.not_training = []
		self.from_scratch_layers = [self.cls_conv]
	
	def __initial__(self):
		for m in self.modules():
			if m not in self.backbone.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
					if m.bias is not None:
						nn.init.constant_(m.bias, 0)
				elif isinstance(m, self.batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		feature = self.maxpool(x_bottom)
		#feature = self.avgpool(feature)
		feature = F.relu(self.conv_fov(feature), inplace=True)
		feature = self.dropout1(feature)
		feature = F.relu(self.conv_fov2(feature), inplace=True)
		feature = self.dropout2(feature)
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
					else:
						groups[0].append(m.weight)

				if m.bias is not None and m.bias.requires_grad:

					if m in self.from_scratch_layers:
						groups[3].append(m.bias)
					else:
						groups[1].append(m.bias)
		return groups
