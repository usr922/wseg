# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import cv2
import numpy as np
import torch
import random
import PIL
from PIL import Image, ImageOps, ImageFilter

class RandomCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):

		h, w = sample['image'].shape[:2]
		ch = min(h, self.output_size[0])
		cw = min(w, self.output_size[1])
		
		h_space = h - self.output_size[0]
		w_space = w - self.output_size[1]

		if w_space > 0:
			cont_left = 0
			img_left = random.randrange(w_space+1)
		else:
			cont_left = random.randrange(-w_space+1)
			img_left = 0

		if h_space > 0:
			cont_top = 0
			img_top = random.randrange(h_space+1)
		else:
			cont_top = random.randrange(-h_space+1)
			img_top = 0

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img_crop = np.zeros((self.output_size[0], self.output_size[1], 3), np.float32)
				img_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 img[img_top:img_top+ch, img_left:img_left+cw]
				#img_crop = img[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = img_crop
			elif 'segmentation' == key:
				seg = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 seg[img_top:img_top+ch, img_left:img_left+cw]
				#seg_crop = seg[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 seg_pseudo[img_top:img_top+ch, img_left:img_left+cw]
				#seg_crop = seg_pseudo[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop
		return sample

class RandomHSV(object):
	"""Generate randomly the image in hsv space."""
	def __init__(self, h_r, s_r, v_r):
		self.h_r = h_r
		self.s_r = s_r
		self.v_r = v_r

	def __call__(self, sample):
		image = sample['image']
		hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		h = hsv[:,:,0].astype(np.int32)
		s = hsv[:,:,1].astype(np.int32)
		v = hsv[:,:,2].astype(np.int32)
		delta_h = random.randint(-self.h_r,self.h_r)
		delta_s = random.randint(-self.s_r,self.s_r)
		delta_v = random.randint(-self.v_r,self.v_r)
		h = (h + delta_h)%180
		s = s + delta_s
		s[s>255] = 255
		s[s<0] = 0
		v = v + delta_v
		v[v>255] = 255
		v[v<0] = 0
		hsv = np.stack([h,s,v], axis=-1).astype(np.uint8)	
		image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
		sample['image'] = image
		return sample

class RandomFlip(object):
	"""Randomly flip image"""
	def __init__(self, threshold):
		self.flip_t = threshold
	def __call__(self, sample):
		if random.random() < self.flip_t:
			key_list = sample.keys()
			for key in key_list:
				if 'image' in key:
					img = sample[key]
					img = np.flip(img, axis=1)
					sample[key] = img
				elif 'segmentation' == key:
					seg = sample[key]
					seg = np.flip(seg, axis=1)
					sample[key] = seg
				elif 'segmentation_pseudo' in key:
					seg_pseudo = sample[key]
					seg_pseudo = np.flip(seg_pseudo, axis=1)
					sample[key] = seg_pseudo
		return sample

class RandomScale(object):
	"""Randomly scale image"""
	def __init__(self, scale_r, is_continuous=False):
		self.scale_r = scale_r
		self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

	def __call__(self, sample):
		row, col, _ = sample['image'].shape
		rand_scale = random.random()*(self.scale_r[1] - self.scale_r[0]) + self.scale_r[0]
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = cv2.resize(img, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
				sample[key] = img
			elif 'segmentation' == key:
				seg = sample[key]
				seg = cv2.resize(seg, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
				sample[key] = seg
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_pseudo = cv2.resize(seg_pseudo, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
				sample[key] = seg_pseudo
		return sample

class ImageNorm(object):
	"""Randomly scale image"""
	def __init__(self, mean=None, std=None):
		self.mean = mean
		self.std = std
	def __call__(self, sample):
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				image = sample[key].astype(np.float32)
				if self.mean is not None and self.std is not None:
					image[...,0] = (image[...,0]/255 - self.mean[0]) / self.std[0]
					image[...,1] = (image[...,1]/255 - self.mean[1]) / self.std[1]
					image[...,2] = (image[...,2]/255 - self.mean[2]) / self.std[2]
				else:
					image /= 255.0
				sample[key] = image
		return sample

class Multiscale(object):
	def __init__(self, rate_list):
		self.rate_list = rate_list

	def __call__(self, sample):
		image = sample['image']
		row, col, _ = image.shape
		image_multiscale = []
		for rate in self.rate_list:
			rescaled_image = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
			sample['image_%f'%rate] = rescaled_image
		return sample


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				image = sample[key].astype(np.float32)
				# swap color axis because
				# numpy image: H x W x C
				# torch image: C X H X W
				image = image.transpose((2,0,1))
				sample[key] = torch.from_numpy(image)
				#sample[key] = torch.from_numpy(image.astype(np.float32)/128.0-1.0)
			elif 'edge' == key:
				edge = sample['edge']
				sample['edge'] = torch.from_numpy(edge.astype(np.float32))
				sample['edge'] = torch.unsqueeze(sample['edge'],0)
			elif 'segmentation' == key:
				segmentation = sample['segmentation']
				sample['segmentation'] = torch.from_numpy(segmentation.astype(np.long))
			elif 'segmentation_pseudo' in key:
				segmentation_pseudo = sample[key]
				sample[key] = torch.from_numpy(segmentation_pseudo.astype(np.float32))
			elif 'segmentation_onehot' == key:
				onehot = sample['segmentation_onehot'].transpose((2,0,1))
				sample['segmentation_onehot'] = torch.from_numpy(onehot.astype(np.float32))
			elif 'category' in key:
				sample[key] = torch.from_numpy(sample[key].astype(np.float32))
			elif 'mask' == key:
				mask = sample['mask']
				sample['mask'] = torch.from_numpy(mask.astype(np.float32))
			elif 'feature' == key:
				feature = sample['feature']
				sample['feature'] = torch.from_numpy(feature.astype(np.float32))
		return sample

