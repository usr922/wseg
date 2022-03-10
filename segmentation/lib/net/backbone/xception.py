""" 
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)
@author: tstandley
Adapted by cadene
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
								  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from net.sync_batchnorm import SynchronizedBatchNorm2d
from utils.registry import BACKBONES

bn_mom = 0.1
__all__ = ['xception']

model_urls = {
	'xception': '/home/wangyude/.cache/torch/checkpoints/xception_pytorch_imagenet.pth'#'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
}

class SeparableConv2d(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,activate_first=True,inplace=True,norm_layer=nn.BatchNorm2d):
		super(SeparableConv2d,self).__init__()
		self.norm_layer = norm_layer
		self.relu0 = nn.ReLU(inplace=inplace)
		self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
		self.bn1 = self.norm_layer(in_channels, momentum=bn_mom)
		self.relu1 = nn.ReLU(inplace=True)
		self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
		self.bn2 = self.norm_layer(out_channels, momentum=bn_mom)
		self.relu2 = nn.ReLU(inplace=True)
		self.activate_first = activate_first
	def forward(self,x):
		if self.activate_first:
			x = self.relu0(x)
		x = self.depthwise(x)
		x = self.bn1(x)
		if not self.activate_first:
			x = self.relu1(x)
		x = self.pointwise(x)
		x = self.bn2(x)
		if not self.activate_first:
			x = self.relu2(x)
		return x


class Block(nn.Module):
	def __init__(self,in_filters,out_filters,strides=1,atrous=None,grow_first=True,activate_first=True,inplace=True,norm_layer=nn.BatchNorm2d):
		super(Block, self).__init__()
		self.norm_layer = norm_layer
		if atrous == None:
			atrous = [1]*3
		elif isinstance(atrous, int):
			atrous_list = [atrous]*3
			atrous = atrous_list
		idx = 0
		self.head_relu = True
		if out_filters != in_filters or strides!=1:
			self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
			self.skipbn = self.norm_layer(out_filters, momentum=bn_mom)
			self.head_relu = False
		else:
			self.skip=None
		
		self.hook_layer = None
		if grow_first:
			filters = out_filters
		else:
			filters = in_filters
		self.sepconv1 = SeparableConv2d(in_filters,filters,3,stride=1,padding=1*atrous[0],dilation=atrous[0],bias=False,activate_first=activate_first,inplace=self.head_relu,norm_layer=self.norm_layer)
		self.sepconv2 = SeparableConv2d(filters,out_filters,3,stride=1,padding=1*atrous[1],dilation=atrous[1],bias=False,activate_first=activate_first,norm_layer=self.norm_layer)
		self.sepconv3 = SeparableConv2d(out_filters,out_filters,3,stride=strides,padding=1*atrous[2],dilation=atrous[2],bias=False,activate_first=activate_first,inplace=inplace,norm_layer=self.norm_layer)

	def forward(self,inp):
		
		if self.skip is not None:
			skip = self.skip(inp)
			skip = self.skipbn(skip)
		else:
			skip = inp

		x = self.sepconv1(inp)
		x = self.sepconv2(x)
		self.hook_layer = x
		x = self.sepconv3(x)

		x+=skip
		return x


class Xception(nn.Module):
	"""
	Xception optimized for the ImageNet dataset, as specified in
	https://arxiv.org/pdf/1610.02357.pdf
	"""
	def __init__(self, os, norm_layer=nn.BatchNorm2d):
		""" Constructor
		Args:
			num_classes: number of classes
		"""
		super(Xception, self).__init__()
		self.norm_layer = norm_layer

		stride_list = None
		if os == 8:
			stride_list = [2,1,1]
		elif os == 16:
			stride_list = [2,2,1]
		else:
			raise ValueError('xception.py: output stride=%d is not supported.'%os) 
		self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
		self.bn1 = self.norm_layer(32, momentum=bn_mom)
		self.relu = nn.ReLU(inplace=True)
		
		self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
		self.bn2 = self.norm_layer(64, momentum=bn_mom)
		#do relu here

		self.block1=Block(64,128,2,norm_layer=self.norm_layer)
		self.block2=Block(128,256,stride_list[0],inplace=False,norm_layer=self.norm_layer)
		self.block3=Block(256,728,stride_list[1],norm_layer=self.norm_layer)

		rate = 16//os
		self.block4=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)
		self.block5=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)
		self.block6=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)
		self.block7=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)

		self.block8=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)
		self.block9=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)
		self.block10=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)
		self.block11=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)

		self.block12=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)
		self.block13=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)
		self.block14=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)
		self.block15=Block(728,728,1,atrous=rate,norm_layer=self.norm_layer)

		self.block16=Block(728,728,1,atrous=[1*rate,1*rate,1*rate],norm_layer=self.norm_layer)
		self.block17=Block(728,728,1,atrous=[1*rate,1*rate,1*rate],norm_layer=self.norm_layer)
		self.block18=Block(728,728,1,atrous=[1*rate,1*rate,1*rate],norm_layer=self.norm_layer)
		self.block19=Block(728,728,1,atrous=[1*rate,1*rate,1*rate],norm_layer=self.norm_layer)
		
		self.block20=Block(728,1024,stride_list[2],atrous=rate,grow_first=False,norm_layer=self.norm_layer)
		#self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

		self.conv3 = SeparableConv2d(1024,1536,3,1,1*rate,dilation=rate,activate_first=False,norm_layer=self.norm_layer)
		# self.bn3 = SynchronizedBatchNorm2d(1536, momentum=bn_mom)

		self.conv4 = SeparableConv2d(1536,1536,3,1,1*rate,dilation=rate,activate_first=False,norm_layer=self.norm_layer)
		# self.bn4 = SynchronizedBatchNorm2d(1536, momentum=bn_mom)

		#do relu here
		self.conv5 = SeparableConv2d(1536,2048,3,1,1*rate,dilation=rate,activate_first=False,norm_layer=self.norm_layer)
		# self.bn5 = SynchronizedBatchNorm2d(2048, momentum=bn_mom)
		self.OUTPUT_DIM = 2048
		self.MIDDLE_DIM = 256

		#------- init weights --------
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, self.norm_layer):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
		#-----------------------------

	def forward(self, input):
		layers = []
		x = self.conv1(input)
		x = self.bn1(x)
		x = self.relu(x)
		#self.layers.append(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		
		x = self.block1(x)
		x = self.block2(x)
		l1 = self.block2.hook_layer
		x = self.block3(x)
		l2 = self.block3.hook_layer
		x = self.block4(x)
		x = self.block5(x)
		x = self.block6(x)
		x = self.block7(x)
		x = self.block8(x)
		x = self.block9(x)
		x = self.block10(x)
		x = self.block11(x)
		x = self.block12(x)
		x = self.block13(x)
		x = self.block14(x)
		x = self.block15(x)
		x = self.block16(x)
		x = self.block17(x)
		x = self.block18(x)
		x = self.block19(x)
		x = self.block20(x)	   
		l3 = self.block20.hook_layer

		x = self.conv3(x)
		# x = self.bn3(x)
		# x = self.relu(x)

		x = self.conv4(x)
		# x = self.bn4(x)
		# x = self.relu(x)
		
		l4 = self.conv5(x)
		# x = self.bn5(x)
		# x = self.relu(x)

		#return layers
		return [l1,l2,l3,l4]

@BACKBONES.register_module
def xception(pretrained=True, os=8, **kwargs):
	model = Xception(os=os)
	if pretrained:
		old_dict = torch.load(model_urls['xception'])
		# old_dict = model_zoo.load_url(model_urls['xception'])
		# for name, weights in old_dict.items():
		#	 if 'pointwise' in name:
		#		 old_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
		model_dict = model.state_dict()
		old_dict = {k: v for k,v in old_dict.items() if ('itr' not in k and 'tmp' not in k and 'track' not in k)}
		model_dict.update(old_dict)
		
		model.load_state_dict(model_dict) 

	return model
