# -*- coding: utf-8 -*-
import math
import numpy as np
import qn

mat = qn.load('data/ipf_log_tpm.pkl')
max_val = np.max(mat)

def sample_z(batchsize,dim):
	temp_norm = np.random.normal(0.0, max_val/10, size=(batchsize, dim))
	temp_poisson = np.random.poisson(1, size=(batchsize, dim))
	return np.abs(temp_norm + temp_poisson).astype('float32')

def sample_x(batchsize,shuffle=True):
	idx = np.random.randint(mat.shape[1], size=batchsize)
	x = mat[:,idx].T
	if shuffle:
		x_shuffled = []
		for row in x:
			indices = np.where(row<3)[0]
			chosen = np.random.choice(indices,size=10,replace=False)
			chosen_vec = row[chosen]
			np.random.shuffle(chosen_vec)
			row[chosen] = chosen_vec
			x_shuffled.append(row)
	else:
		x_shuffled = x
	return np.array(x_shuffled).astype('float32')
