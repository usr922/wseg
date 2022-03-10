# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import multiprocessing
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import *
from utils.imutils import *
from utils.registry import DATASETS

#@DATASETS.register_module
class BaseDataset(Dataset):
	def __init__(self, cfg, period, transform='none'):
		super(BaseDataset, self).__init__()
		self.cfg = cfg
		self.period = period
		self.transform = transform
		if 'train' not in self.period:
			assert self.transform == 'none'
		self.num_categories = None
		self.totensor = ToTensor()
		self.imagenorm = ImageNorm(cfg.DATA_MEAN, cfg.DATA_STD)
		
		if self.transform != 'none':
			if cfg.DATA_RANDOMCROP > 0:
				self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
			if cfg.DATA_RANDOMSCALE != 1:
				self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
			if cfg.DATA_RANDOMFLIP > 0:
				self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
			if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
				self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
		else:
			self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)


	def __getitem__(self, idx):
		sample = self.__sample_generate__(idx)

		if 'segmentation' in sample.keys():
			sample['mask'] = sample['segmentation'] < self.num_categories
			t = sample['segmentation'].copy()
			t[t >= self.num_categories] = 0
			sample['segmentation_onehot']=onehot(t,self.num_categories)
		return self.totensor(sample)

	def __sample_generate__(self, idx, split_idx=0):
		name = self.load_name(idx)
		image = self.load_image(idx)
		r,c,_ = image.shape
		sample = {'image': image, 'name': name, 'row': r, 'col': c}

		if 'test' in self.period:
			return self.__transform__(sample)
		elif self.cfg.DATA_PSEUDO_GT and idx>=split_idx and 'train' in self.period:
			segmentation = self.load_pseudo_segmentation(idx)
		else:
			segmentation = self.load_segmentation(idx)
		sample['segmentation'] = segmentation
		t = sample['segmentation'].copy()
		t[t >= self.num_categories] = 0
		sample['category'] = seg2cls(t,self.num_categories)
		sample['category_copypaste'] = np.zeros(sample['category'].shape)

		#if self.transform == 'none' and self.cfg.DATA_FEATURE_DIR:
		#	feature = self.load_feature(idx)
		#	sample['feature'] = feature
		return self.__transform__(sample)

	def __transform__(self, sample):
		if self.transform == 'weak':
			sample = self.__weak_augment__(sample)
		elif self.transform == 'strong':
			sample = self.__strong_augment__(sample)
		else:
			sample = self.imagenorm(sample)
			sample = self.multiscale(sample)
		return sample

	def __weak_augment__(self, sample):
		if self.cfg.DATA_RANDOM_H>0 or self.cfg.DATA_RANDOM_S>0 or self.cfg.DATA_RANDOM_V>0:
			sample = self.randomhsv(sample)
		if self.cfg.DATA_RANDOMFLIP > 0:
			sample = self.randomflip(sample)
		if self.cfg.DATA_RANDOMSCALE != 1:
			sample = self.randomscale(sample)
		sample = self.imagenorm(sample)
		if self.cfg.DATA_RANDOMCROP > 0:
			sample = self.randomcrop(sample)
		return sample

	def __strong_augment__(self, sample):
		raise NotImplementedError
		
	def __len__(self):
		raise NotImplementedError

	def load_name(self, idx):
		raise NotImplementedError	

	def load_image(self, idx):
		raise NotImplementedError	

	def load_segmentation(self, idx):
		raise NotImplementedError

	def load_pseudo_segmentation(self, idx):
		raise NotImplementedError

	def load_feature(self, idx):
		raise NotImplementedError
		
	def save_result(self, result_list, model_id):
		raise NotImplementedError	

	def save_pseudo_gt(self, result_list, level=None):
		raise NotImplementedError

	def do_python_eval(self, model_id):
		raise NotImplementedError
