# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os, glob
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
from collections import namedtuple
from utils.registry import DATASETS
from datasets.BaseDataset import BaseDataset

Label = namedtuple( 'Label', ['name','id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color',])
@DATASETS.register_module
class CityscapesDataset(BaseDataset):
	def __init__(self, cfg, period, transform=False):
		super(CityscapesDataset, self).__init__(cfg, period, transform)
		self.root_dir = os.path.join(cfg.ROOT_DIR,'data','cityscapes')
		self.dataset_dir = self.root_dir
		self.rst_dir = os.path.join(self.dataset_dir,'results')
		self.img_dir = os.path.join(self.dataset_dir, 'leftImg8bit',self.period)
		self.seg_dir = os.path.join(self.dataset_dir, 'gtFine',self.period)
		self.img_extra_dir = os.path.join(self.dataset_dir, 'leftImg8bit', 'train_extra')
		self.seg_extra_dir = os.path.join(self.dataset_dir, 'gtCoarse', 'train_extra')
		if cfg.DATA_PSEUDO_GT:
			self.pseudo_gt_dir = cfg.DATA_PSEUDO_GT
		else:
			self.pseudo_gt_dir = os.path.join(self.root_dir,'pseudo_gt')
		
		searchFine = os.path.join(self.img_dir,'*', '*_*_*_leftImg8bit.png' )
		filesFine = glob.glob(searchFine)
		filesFine.sort()
		self.name_list = []
		for file in filesFine:
			name = file.replace('%s/'%self.img_dir,'')
			name = name.replace('_leftImg8bit.png','')
			self.name_list.append(name)

		if cfg.DATA_AUG:
			searchCoarse = os.path.join(self.img_extra_dir, '*', '*_*_*_leftImg8bit.png')
			filesCoarse = glob.glob(searchCoarse)
			filesCoarse.sort()
			for file in filesCoarse:
				name = file.replace('%s/'%self.img_extra_dir,'')
				name = name.replace('_leftImg8bit.png','')
				self.name_list.append(name)

		self.categories= [
			#	 name	 		id	trainId   category	catId	 hasInstances	ignoreInEval	color
			Label(  'unlabeled',		0 ,	255 , 	 'void',	0,	 False,		True,		 (  0,  0,  0) ),
			Label(  'ego vehicle',		1 ,	255 ,	 'void',	0,	 False,		True,		 (  0,  0,  0) ),
			Label(  'rectification border', 2 ,	255 ,	 'void',	0,	 False,		True,		 (  0,  0,  0) ),
			Label(  'out of roi',		3 ,	255 ,	 'void',	0,	 False,		True,		 (  0,  0,  0) ),
			Label(  'static',		4 ,	255 ,	 'void',	0,	 False,		True,		 (  0,  0,  0) ),
			Label(  'dynamic',		5 ,	255 ,	 'void',	0,	 False,		True,		 (111, 74,  0) ),
			Label(  'ground',		6 ,	255 ,	 'void',	0,	 False,		True,		 ( 81,  0, 81) ),
			Label(  'road',			7 ,	0 ,	 'flat',	1,	 False,		False,		 (128, 64,128) ),
			Label(  'sidewalk',		8 ,	1 , 	 'flat',	1,	 False,		False,		 (244, 35,232) ),
			Label(  'parking',		9 ,	255 , 	 'flat',	1,	 False,		True,		 (250,170,160) ),
			Label(  'rail track',		10 ,	255 , 	 'flat',	1,	 False,		True,		 (230,150,140) ),
			Label(  'building',		11 ,	2 , 	 'construction',2,	 False,		False,		 ( 70, 70, 70) ),
			Label(  'wall',			12 ,	3 , 	 'construction',2,	 False,		False,		 (102,102,156) ),
			Label(  'fence',		13 ,	4 , 	 'construction',2,	 False,		False,		 (190,153,153) ),
			Label(  'guard rail',		14 ,	255 , 	 'construction',2,	 False,		True,		 (180,165,180) ),
			Label(  'bridge',		15 ,	255 , 	 'construction',2,	 False,		True,		 (150,100,100) ),
			Label(  'tunnel',		16 ,	255 , 	 'construction',2,	 False,		True,		 (150,120, 90) ),
			Label(  'pole',			17 ,	5 , 	 'object',	3,	 False,		False,		 (153,153,153) ),
			Label(  'polegroup',		18 ,	255 , 	 'object',	3,	 False,		True,		 (153,153,153) ),
			Label(  'traffic light',	19 ,	6 , 	 'object',	3,	 False,		False,		 (250,170, 30) ),
			Label(  'traffic sign',		20 ,	7 , 	 'object',	3,	 False,		False,		 (220,220,  0) ),
			Label(  'vegetation',		21 ,	8 , 	 'nature',	4,	 False,		False,		 (107,142, 35) ),
			Label(  'terrain',		22 ,	9 , 	 'nature',	4,	 False,		False,		 (152,251,152) ),
			Label(  'sky',			23 ,	10 , 	 'sky',		5,	 False,		False,		 ( 70,130,180) ),
			Label(  'person',		24 ,	11 , 	 'human',	6,	 True ,		False,		 (220, 20, 60) ),
			Label(  'rider',		25 ,	12 , 	 'human',	6,	 True ,		False,		 (255,  0,  0) ),
			Label(  'car',			26 ,	13 , 	 'vehicle',	7,	 True ,		False,		 (  0,  0,142) ),
			Label(  'truck',		27 ,	14 , 	 'vehicle',	7,	 True ,		False,		 (  0,  0, 70) ),
			Label(  'bus',			28 ,	15 , 	 'vehicle',	7,	 True ,		False,		 (  0, 60,100) ),
			Label(  'caravan',		29 ,	255 , 	 'vehicle',	7,	 True ,		True,		 (  0,  0, 90) ),
			Label(  'trailer',		30 ,	255 , 	 'vehicle',	7,	 True ,		True,		 (  0,  0,110) ),
			Label(  'train',		31 ,	16 , 	 'vehicle',	7,	 True ,		False,		 (  0, 80,100) ),
			Label(  'motorcycle',		32 ,	17 , 	 'vehicle',	7,	 True ,		False,		 (  0,  0,230) ),
			Label(  'bicycle',		33 ,	18 , 	 'vehicle',	7,	 True ,		False,		 (119, 11, 32) ),
			Label(  'license plate',	-1 ,	-1 ,	 'vehicle',	7,	 False,		True,		 (  0,  0,142) ),
		]
		self.id2label = {label.id: label for label in self.categories}
		self.trainId2label = {label.trainId : label for label in reversed(self.categories)}
		self.num_categories = 19
		assert self.num_categories == self.cfg.MODEL_NUM_CLASSES
		
	def __len__(self):
		return len(self.name_list)

	def load_name(self, idx):
		return self.name_list[idx]

	def load_image(self, idx):
		name = self.name_list[idx]
		img_file = os.path.join(self.img_dir, name + '_leftImg8bit.png')
		if not os.path.exists(img_file):
			img_file = os.path.join(self.img_extra_dir, name + '_leftImg8bit.png')
		image = cv2.imread(img_file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image

	def __id2trainid__(self, seg):
		for label in self.categories:
			seg[seg == label.id] = label.trainId
		#seg[seg==-1] = 34
		return seg

	def load_segmentation(self, idx):
		name = self.name_list[idx]
		seg_file = os.path.join(self.seg_dir, name + '_gtFine_labelIds.png')
		if not os.path.exists(seg_file):
			seg_file = os.path.join(self.seg_extra_dir, name + '_gtCoarse_labelIds.png')
		segmentation = np.array(Image.open(seg_file))
		return self.__id2trainid__(segmentation)

	def load_pseudo_segmentation(self, idx):
		name = self.name_list[idx].split('/')[1]
		seg_file = os.path.join(self.seg_dir, name + '.png')
		segmentation = np.array(Image.open(seg_file))
		return self.__id2trainid__(segmentation)
		
	def save_pseudo_gt(self, result_list, level=None):
		folder_path = self.pseudo_gt_dir
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		for sample in result_list:
			name = sample['name'].split('/')
			file_path = os.path.join(folder_path, '%s.png'%name[1])
			cv2.imwrite(file_path, sample['predict'])
			print('%s saved'%(file_path))

	def do_python_eval(self, model_id):
		raise NotImplementedError


	def label2colormap(self, label, id_version='trainid'):
		m = label.astype(np.uint8)
		r,c = m.shape
		cmap = np.zeros((r,c,3), dtype=np.uint8)
		if id_version == 'id':
			for k in self.id2label.keys():
				cmap[m == k] = self.id2label[k].color
		elif id_version == 'trainid':
			for k in self.trainId2label.keys():
				cmap[m == k] = self.trainId2label[k].color
		return cmap

	def trainid2id(self, label):
		label_id = label.copy()
		for k in self.trainId2label.keys():
			label_id[label == k] = self.trainId2label[k].id 
		return label_id
	
	def save_result(self, result_list, model_id):
		"""Save test results

		Args:
			result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

		"""
		i = 1
		folder_path = self.rst_dir
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		for sample in result_list:
			name = sample['name'].split('/')
			file_path = os.path.join(folder_path, '%s.png'%name[1])
			# predict_color = self.label2colormap(sample['predict'])
			# p = self.__coco2voc(sample['predict'])
			cv2.imwrite(file_path, sample['predict'])
			print('[%d/%d] %s saved'%(i,len(result_list),file_path))
			i+=1
	
	def do_cityscapesscripts_eval(self):
		import subprocess
		path = self.root_dir
		cmd = 'cd {} && '.format(path)
		cmd += 'python cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py'

		print('start subprocess for cityscapesscripts evaluation...')
		print(cmd)
		subprocess.call(cmd, shell=True)


