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

@NETS.register_module
class deeplabv3plus(nn.Module):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3plus, self).__init__()
		self.cfg = cfg
		self.batchnorm = batchnorm
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=cfg.MODEL_BACKBONE_PRETRAIN, norm_layer=self.batchnorm, **kwargs)
		input_channel = self.backbone.OUTPUT_DIM
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=[0, 6, 12, 18],
				bn_mom = cfg.TRAIN_BN_MOM,
				has_global = cfg.MODEL_ASPP_HASGLOBAL,
				batchnorm = self.batchnorm)
		#self.dropout1 = nn.Dropout(0.5)

		indim = self.backbone.MIDDLE_DIM
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, 3, 1, padding=1, bias=False),
				batchnorm(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),		
		)		
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
				batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),
				#nn.Dropout(0.5),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
				batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),
				#nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		for m in self.modules():
			if m not in self.backbone.modules():
		#		if isinstance(m, nn.Conv2d):
		#			nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if isinstance(m, batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		if cfg.MODEL_FREEZEBN:
			self.freeze_bn()

	def forward(self, x, getf=False, interpolate=True):
		N,C,H,W = x.size()
		l1, l2, l3, l4 = self.backbone(x)
		feature_aspp = self.aspp(l4)
		#feature_aspp = self.dropout1(feature_aspp)

		feature_shallow = self.shortcut_conv(l1)
		n,c,h,w = feature_shallow.size()
		feature_aspp = F.interpolate(feature_aspp,(h,w),mode='bilinear',align_corners=True)

		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		feature = self.cat_conv(feature_cat) 
		result = self.cls_conv(feature)
		result = F.interpolate(result, (H,W), mode='bilinear',align_corners=True)

		if getf:
			if interpolate:
				feature = F.interpolate(feature, (H,W), mode='bilinear', align_corners=True)
			return result, feature
		else:
			return result

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, self.batchnorm):
				m.eval()
	def unfreeze_bn(self):
		for m in self.modules():
			if isinstance(m, self.batchnorm):
				m.train()

@NETS.register_module
class deeplabv3plus2d(deeplabv3plus):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3plus2d, self).__init__(cfg, batchnorm=batchnorm, **kwargs)
		self.compress_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, 2, 1, 1, padding=0, bias=False)
		self.cls_conv = nn.Conv2d(2, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0, bias=False)
		for m in self.modules():
			if m not in self.backbone.modules():
				if isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if isinstance(m, batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		if cfg.MODEL_FREEZEBN:
			self.freeze_bn()

	def forward(self, x, getf=False, interpolate=True):
		N,C,H,W = x.size()
		l1, l2, l3, l4 = self.backbone(x)
		feature_aspp = self.aspp(l4)
		#feature_aspp = self.dropout1(feature_aspp)

		feature_shallow = self.shortcut_conv(l1)
		n,c,h,w = feature_shallow.size()
		feature_aspp = F.interpolate(feature_aspp,(h,w),mode='bilinear',align_corners=True)

		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		feature = self.cat_conv(feature_cat) 
		feature = self.compress_conv(feature)
		result = self.cls_conv(feature)
		result = F.interpolate(result, (H,W), mode='bilinear',align_corners=True)

		if getf:
			if interpolate:
				feature = F.interpolate(feature, (H,W), mode='bilinear', align_corners=True)
			return result, feature
		else:
			return result

@NETS.register_module
class deeplabv3plusInsNorm(deeplabv3plus):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3plusInsNorm, self).__init__(cfg, batchnorm, **kwargs)
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
				nn.InstanceNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=False),
				nn.InstanceNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),
		)
		for m in self.modules():
			if m not in self.backbone.modules():
				if isinstance(m, (batchnorm, nn.InstanceNorm2d)):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		if cfg.MODEL_FREEZEBN:
			self.freeze_bn()

@NETS.register_module
class deeplabv3plusAux(deeplabv3plus):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3plusAux, self).__init__(cfg, batchnorm, **kwargs)
		input_channel = self.backbone.OUTPUT_DIM
		self.seghead2 = nn.Sequential(
				nn.Conv2d(input_channel//4, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=False),
				batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		)
		self.seghead3 = nn.Sequential(
				nn.Conv2d(input_channel//2, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=False),
				batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		)
		self.seghead4 = nn.Sequential(
				nn.Conv2d(input_channel, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=False),
				batchnorm(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM, affine=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		)
		#self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0, bias=False)
		for m in self.modules():
			if m not in self.backbone.modules():
		#		if isinstance(m, nn.Conv2d):
		#			nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if isinstance(m, batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		if cfg.MODEL_FREEZEBN:
			self.freeze_bn()

	def forward(self, x, getf=False, interpolate=True):
		N,C,H,W = x.size()
		l1, l2, l3, l4 = self.backbone(x)
		feature_aspp = self.aspp(l4)

		feature_shallow = self.shortcut_conv(l1)
		n,c,h,w = feature_shallow.size()
		feature_aspp = F.interpolate(feature_aspp,(h,w),mode='bilinear',align_corners=True)

		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		feature = self.cat_conv(feature_cat) 
		result = self.cls_conv(feature)
		result = F.interpolate(result, (H,W), mode='bilinear',align_corners=True)

		seg2 = F.interpolate(self.seghead2(l2), (H,W), mode='bilinear', align_corners=True)
		seg3 = F.interpolate(self.seghead3(l3), (H,W), mode='bilinear', align_corners=True)
		seg4 = F.interpolate(self.seghead4(l4), (H,W), mode='bilinear', align_corners=True)
		
		if getf:
			if interpolate:
				feature = F.interpolate(feature, (H,W), mode='bilinear', align_corners=True)
			return [result, seg2, seg3, seg4], feature
		else:
			return [result, seg2, seg3, seg4]

	def orth_init(self):
		self.cls_conv.weight = torch.nn.Parameter(torch.eye(n=self.cfg.MODEL_NUM_CLASSES, m=self.cfg.MODEL_ASPP_OUTDIM).unsqueeze(-1).unsqueeze(-1))
		self.seghead2[-1].weight = torch.nn.Parameter(torch.eye(n=self.cfg.MODEL_NUM_CLASSES, m=self.cfg.MODEL_ASPP_OUTDIM).unsqueeze(-1).unsqueeze(-1))
		self.seghead3[-1].weight = torch.nn.Parameter(torch.eye(n=self.cfg.MODEL_NUM_CLASSES, m=self.cfg.MODEL_ASPP_OUTDIM).unsqueeze(-1).unsqueeze(-1))
		self.seghead4[-1].weight = torch.nn.Parameter(torch.eye(n=self.cfg.MODEL_NUM_CLASSES, m=self.cfg.MODEL_ASPP_OUTDIM).unsqueeze(-1).unsqueeze(-1))
		print('deeplabv3plusAux orth_init() finished')

	def orth_reg(self):
		module_list = [self.cls_conv, self.seghead2[-1], self.seghead3[-1], self.seghead4[-1]]
		loss_reg = 0
		for m in module_list:
			w = m.weight.squeeze(-1).squeeze(-1)
			w_norm = torch.norm(w, dim=1, keepdim=True)
			w = w/w_norm
			matrix = torch.matmul(w, w.transpose(0,1))
			loss_reg += torch.mean(matrix*(1-torch.eye(self.cfg.MODEL_NUM_CLASSES).to(0)))
		return loss_reg


@NETS.register_module
class deeplabv3plusAuxSigmoid(deeplabv3plusAux):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3plusAuxSigmoid, self).__init__(cfg, batchnorm, **kwargs)
		for m in self.modules():
			if m not in self.backbone.modules() and isinstance(m, nn.ReLU):
				m = nn.Sigmoid()

@NETS.register_module
class deeplabv3plusAuxReLUSigmoid(deeplabv3plusAux):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3plusAuxReLUSigmoid, self).__init__(cfg, batchnorm, **kwargs)
		for m in self.modules():
			if isinstance(m, nn.ReLU):
				m = nn.Sequential(
					nn.ReLU(inplace=True),
					nn.Sigmoid()
				)

@NETS.register_module
class deeplabv3plusNorm(deeplabv3plus):
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3plusNorm, self).__init__(cfg, batchnorm, **kwargs)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0, bias=False)

	def forward(self, x, getf=False, interpolate=True):
		N,C,H,W = x.size()
		l1, l2, l3, l4 = self.backbone(x)
		feature_aspp = self.aspp(l4)
		#feature_aspp = self.dropout1(feature_aspp)

		feature_shallow = self.shortcut_conv(l1)
		n,c,h,w = feature_shallow.size()
		feature_aspp = F.interpolate(feature_aspp,(h,w),mode='bilinear',align_corners=True)

		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		feature = self.cat_conv(feature_cat) 
		feature_norm = torch.norm(feature, dim=1, keepdim=True).detach()
		feature = feature/feature_norm
		conv_norm = torch.norm(self.cls_conv.weight, dim=1, keepdim=True).detach()
		conv_norm = conv_norm.permute(1,0,2,3)
		result = self.cls_conv(feature)/conv_norm
		
		result = F.interpolate(result, (H,W), mode='bilinear',align_corners=True)

		if getf:
			if interpolate:
				feature = F.interpolate(feature, (H,W), mode='bilinear', align_corners=True)
			return result, feature
		else:
			return result
