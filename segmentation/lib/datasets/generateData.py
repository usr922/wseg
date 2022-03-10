# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

#import torch
#import torch.nn as nn
#from datasets.VOCDataset import VOCDataset, Semi_VOCDataset, VOCSuperPixelDataset
#from datasets.COCODataset import COCOSmtDataset, COCOInsDataset
#from datasets.ADE20KDataset import ADE20KDataset
#from datasets.ContextDataset import ContextDataset
#from datasets.CityscapesDataset import CityscapesDataset
#from datasets.CityscapesDemoDataset import CityscapesDemoDataset
from utils.registry import DATASETS

#def generate_dataset(dataset_name, cfg, data_period, data_aug=False, aug_period=None):
#	if dataset_name == 'voc2012' or dataset_name == 'VOC2012':
#		return VOCDataset('VOC2012', cfg, data_period, data_aug=data_aug, aug_period=aug_period)
#	elif dataset_name == 'semi-voc2012' or dataset_name == 'Semi-VOC2012':
#		return Semi_VOCDataset('VOC2012', cfg, data_period, data_aug=data_aug, aug_period=aug_period)
#	elif dataset_name == 'vocsp2012' or dataset_name == 'VOCSuperPixel2012':
#		return VOCSuperPixelDataset('VOC2012', cfg, data_period, data_aug=data_aug, aug_period=aug_period)
#	elif dataset_name == 'coco2017smt' or dataset_name == 'COCO2017Smt':
#		return COCODataset('COCO2017', cfg, data_period)
#	elif dataset_name == 'ade20k' or dataset_name == 'ADE20K':
#		return ADE20KDataset('ADE20K', cfg, data_period)
#	elif dataset_name == 'context' or dataset_name == 'Context':
#		return ContextDataset('Context', cfg, data_period)
#	elif dataset_name == 'cityscapes' or dataset_name == 'Cityscapes':
#		return CityscapesDataset('Cityscapes', cfg, data_period, data_aug=data_aug, aug_period=aug_period)
#	elif dataset_name == 'cityscapesdemo' or dataset_name == 'CityscapesDemo':
#		return CityscapesDemoDataset('CityscapesDemo', cfg, data_period)
#	else:
#		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)

def generate_dataset(cfg, **kwargs):
	dataset = DATASETS.get(cfg.DATA_NAME)(cfg, **kwargs)
	return dataset
