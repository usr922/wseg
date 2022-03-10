import numpy as np
import cv2

def pseudo_erode(label, num, t=1):
	label_onehot = onehot(label, num)
	k = np.ones((15,15),np.uint8)
	e = cv2.erode(label_onehot, k, t)
	m = (e != label_onehot)
	m = np.max(m, axis=2)
	label[m] = 255
	return label
	

def onehot(label, num):
	num = int(num)
	m = label.astype(np.int32)
	one_hot = np.eye(num)[m]
	return one_hot

def seg2cls(label, num):
	cls = np.zeros(num)
	index = np.unique(label)
	cls[index] = 1
	#cls[0] = 0
	cls = cls.reshape((num,1,1))
	return cls

def gamma_correction(img):
	gamma = np.mean(img)/128.0
	lookUpTable = np.empty((1,256), np.uint8)
	for i in range(256):
		lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
	res_img = cv2.LUT(img, lookUpTable)
	return res_img

def img_denorm(inputs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), mul=True):
	inputs = np.ascontiguousarray(inputs)
	if inputs.ndim == 3:
		inputs[0,:,:] = (inputs[0,:,:]*std[0] + mean[0])
		inputs[1,:,:] = (inputs[1,:,:]*std[1] + mean[1])
		inputs[2,:,:] = (inputs[2,:,:]*std[2] + mean[2])
	elif inputs.ndim == 4:
		n = inputs.shape[0]	
		for i in range(n):
			inputs[i,0,:,:] = (inputs[i,0,:,:]*std[0] + mean[0])
			inputs[i,1,:,:] = (inputs[i,1,:,:]*std[1] + mean[1])
			inputs[i,2,:,:] = (inputs[i,2,:,:]*std[2] + mean[2])
		
	if mul:
		inputs = inputs*255
		inputs[inputs > 255] = 255
		inputs[inputs < 0] = 0
		inputs = inputs.astype(np.uint8)
	else:
		inputs[inputs > 1] = 1
		inputs[inputs < 0] = 0
	return inputs
