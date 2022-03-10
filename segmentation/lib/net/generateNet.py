# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

#from net.deeplabv3plus import deeplabv3plus
#from net.deeplabv3 import deeplabv3, deeplabv3_noise, deeplabv3_feature, deeplabv3_glore
#from net.deeplabv2 import deeplabv2, deeplabv2_caffe
#from net.deeplabv1 import deeplabv1, deeplabv1_caffe
#from net.clsnet import ClsNet
#from net.fcn import FCN
#from net.DFANet import DFANet
from utils.registry import NETS

def generate_net(cfg, **kwargs):
	net = NETS.get(cfg.MODEL_NAME)(cfg, **kwargs)
	return net
#def generate_net(cfg):
#	if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
#		return deeplabv3plus(cfg)
#	elif cfg.MODEL_NAME == 'deeplabv3':
#		return deeplabv3(cfg)
#	elif cfg.MODEL_NAME == 'deeplabv2':
#		return deeplabv2(cfg)
#	elif cfg.MODEL_NAME == 'deeplabv1':
#		return deeplabv1(cfg)
#	elif cfg.MODEL_NAME == 'deeplabv1_caffe':
#		return deeplabv1_caffe(cfg)
#	elif cfg.MODEL_NAME == 'deeplabv2_caffe':
#		return deeplabv2_caffe(cfg)
#	elif cfg.MODEL_NAME == 'clsnet' or cfg.MODEL_NAME == 'ClsNet':
#		return ClsNet(cfg)
#	elif cfg.MODEL_NAME == 'fcn' or cfg.MODEL_NAME == 'FCN':
#		return FCN(cfg)
#	elif cfg.MODEL_NAME == 'DFANet' or cfg.MODEL_NAME == 'dfanet':
#		return DFANet(cfg)	
#	else:
#		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
