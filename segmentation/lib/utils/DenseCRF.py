import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def dense_crf(probs, img=None, n_classes=21, n_iters=1, scale_factor=1):
	#probs = np.transpose(probs,(1,2,0)).copy(order='C')
	c,h,w = probs.shape

	if img is not None:
		assert(img.shape[1:3] == (h, w))
		img = np.transpose(img,(1,2,0)).copy(order='C')

	#probs = probs.transpose(2, 0, 1).copy(order='C') # Need a contiguous array.

	d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
	
	unary = unary_from_softmax(probs)
	unary = np.ascontiguousarray(unary)
	d.setUnaryEnergy(unary)
	d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
	#d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
	d.addPairwiseBilateral(sxy=32/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
	Q = d.inference(n_iters)

#	U = -np.log(probs) # Unary potential.
#	U = U.reshape((n_classes, -1)) # Needs to be flat.
#	d.setUnaryEnergy(U)
#	d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
#			kernel=kernel_gaussian, normalization=normalisation_gaussian)
#	if img is not None:
#		assert(img.shape[1:3] == (h, w))
#		img = np.transpose(img,(1,2,0)).copy(order='C')
#		d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
#				kernel=kernel_bilateral, normalization=normalisation_bilateral,
#				srgb=srgb_bilateral, rgbim=img)
#	Q = d.inference(n_iters)
	preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w))
	#return np.expand_dims(preds, 0)
	return preds

def pro_crf(p, img, itr):
	C, H, W = p.shape
	p_bg = 1-p
	for i in range(C):
		cat = np.concatenate([p[i,:,:], p_bg[i,:,:]], axis=0)
		crf_pro = dense_crf(cat, img.astype(np.uint8), n_classes=C, n_iters=itr)
		p[i,:,:] = crf_pro[0]
	return p
