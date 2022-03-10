# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import json
import torch
from torch.utils.data import Dataset
import cv2
#from scipy.misc import imread
import numpy as np
from datasets.transform import *
from datasets.metric import AverageMeter, accuracy, intersectionAndUnion
from utils.registry import DATASETS
from datasets.BaseDataset import BaseDataset

@DATASETS.register_module
class ADE20KDataset(BaseDataset):
	def __init__(self, cfg, period, transform='none'):
		super(ADE20KDataset, self).__init__(cfg, period, transform)
		assert(self.period != 'test')
		self.root_dir = os.path.join(cfg.ROOT_DIR,'data','ADEChallengeData2016')
		self.dataset_dir = self.root_dir
		self.img_dir = os.path.join(self.dataset_dir, 'images')
		self.seg_dir = os.path.join(self.dataset_dir, 'annotations')
		self.rst_dir = os.path.join(self.dataset_dir,'result')
		if cfg.DATA_PSEUDO_GT:
			self.pseudo_gt_dir = cfg.DATA_PSEUDO_GT
		else:
			self.pseudo_gt_dir = os.path.join(self.root_dir,'pseudo_gt')
		self.num_categories = 151
		assert(self.num_categories == self.cfg.MODEL_NUM_CLASSES)
		if self.period == 'train':
			self.name_list = ['ADE_train_%08d'%(i+1) for i in range(20210)]
		elif self.period == 'val':
			self.name_list = ['ADE_val_%08d'%(i+1) for i in range(2000)]
		else:
			raise ValueError('self.period is not \'train\' or \'val\'')

	def __len__(self):
		return len(self.name_list)

	def load_name(self, idx):
		name = self.name_list[idx]
		return name

	def load_image(self, idx):
		name = self.name_list[idx]
		if self.period == 'train':
			img_file = os.path.join(self.img_dir,'training',name+'.jpg')
		elif self.period == 'val':
			img_file = os.path.join(self.img_dir,'validation',name+'.jpg')
		else:
			raise ValueError('self.period is not \'train\' or \'val\'')
		image = cv2.imread(img_file)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image_rgb

	def load_segmentation(self, idx):
		name = self.name_list[idx]
		if self.period == 'train':
			seg_file = os.path.join(self.seg_dir,'training',name+'.png')
		elif self.period == 'val':
			seg_file = os.path.join(self.seg_dir,'validation',name+'.png')
		else:
			raise ValueError('self.period is not \'train\' or \'val\'')
		segmentation = np.array(Image.open(seg_file))
		assert np.min(segmentation)>=0
		assert np.max(segmentation)<self.num_categories
		#seg[seg>=self.cfg.MODEL_NUM_CLASSES] = 0
		#seg += 1
		return segmentation

	def load_pseudo_segmentation(self, idx):
		name = self.name_list[idx]
		seg_file = os.path.join(self.pseudo_gt_dir,name+'.png')
		segmentation = np.array(Image.open(seg_file))
		return segmentation
		
	def save_pseudo_gt(self, result_list):
		"""Save pseudo gt

		Args:
			result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

		"""
		folder_path = self.pseudo_gt_dir
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		for sample in result_list:
			file_path = os.path.join(folder_path, '%s.png'%(sample['name']))
			cv2.imwrite(file_path, sample['predict'])
			print('%s saved'%(file_path))
 
	def label2colormap(self, label):
		m = label.astype(np.uint8)
		r,c = m.shape
		cmap = np.zeros((r,c,3), dtype=np.uint8)
		cmap[:,:,0] = (m&1)<<7 | (m&8)<<3 | (m&64)>>1
		cmap[:,:,1] = (m&2)<<6 | (m&16)<<2 | (m&128)>>2
		cmap[:,:,2] = (m&4)<<5 | (m&32)<<1
		return cmap

	def save_result(self, result_list, model_id):
		folder_path = os.path.join(self.rst_dir,'%s'%model_id)
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		for sample in result_list:
			file_path = os.path.join(folder_path,'%s.png'%sample['name'])
			'''

			ATTENTION!!!

			predict label start from 0 or -1 ?????

			DO NOT have operation here!!!


			'''
			cv2.imwrite(file_path, sample['predict'])

	def do_python_eval(self, model_id):
		folder_path = os.path.join(self.rst_dir,'%s'%model_id)

		acc_meter = AverageMeter()
		intersection_meter = AverageMeter()
		union_meter = AverageMeter()
		for name in self.name_list:
			predict_path = os.path.join(folder_path,'%s.png'%name)
			if 'train' in name:
				label_path = os.path.join(self.seg_dir, 'training', name+'.png')
			elif 'val' in name:
				label_path = os.path.join(self.seg_dir, 'validation', name+'.png')
			else:
				raise ValueError('self.period is not \'train\' or \'val\'')
			
			#predict = imread(predict_path)
			#label = imread(label_path)
			predict = np.array(Image.open(predict_path))
			segmentation = np.array(Image.open(label_pth))

			acc, pix = accuracy(predict, label)
			intersection, union = intersectionAndUnion(predict, label, self.num_categories)
			acc_meter.update(acc, pix)
			intersection_meter.update(intersection)
			union_meter.update(union)

		iou = intersection_meter.sum / (union_meter.sum + 1e-10)
		loglist = {}
		for i, _iou in enumerate(iou):
			print('class [{}], IoU: {}'.format(i, _iou))
			loglist['class[{}]'.format(i)] = _iou
		print('[Eval Summary]:')
		print('Mean IoU: {:.4}, Accuracy: {:.2f}%'.format(iou.mean(), acc_meter.average()*100))
		loglist['mIoU'] = iou.mean()
		loglist['accuracy'] = acc_meter.average()*100
		return loglist
