# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import os
import sys
import shutil

class Configuration():
	def __init__(self, config_dict, clear=True):
		self.__dict__ = config_dict
		self.clear = clear
		self.__check()
		
	def __check(self):
		if not torch.cuda.is_available():
			raise ValueError('config.py: cuda is not avalable')
		if self.GPUS == 0:
			raise ValueError('config.py: the number of GPU is 0')
		if self.GPUS != torch.cuda.device_count():
			raise ValueError('config.py: GPU number is not matched')
		if not os.path.isdir(self.LOG_DIR):
			os.makedirs(self.LOG_DIR)
		elif self.clear:
			shutil.rmtree(self.LOG_DIR)
			os.mkdir(self.LOG_DIR)
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)





