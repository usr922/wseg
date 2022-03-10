import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.Module):
	"""
	Reference:
		Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
	"""
	def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
		super(PPM, self).__init__()

		self.stages = []
		self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
		self.bottleneck = nn.Sequential(
			nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
			norm_layer(out_features),
			nn.ReLU(),
			nn.Dropout2d(0.1)
			)

	def _make_stage(self, features, out_features, size, norm_layer):
		prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
		conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
		bn = norm_layer(out_features)
		return nn.Sequential(prior, conv, bn)

	def forward(self, feats):
		h, w = feats.size(2), feats.size(3)
		priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
		bottle = self.bottleneck(torch.cat(priors, 1))
		return bottle
