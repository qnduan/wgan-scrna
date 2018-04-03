# -*- coding: utf-8 -*-
import sampler, pylab, os
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
# from model import discriminator_params, generator_params, gan
#from args import args
# sns.set(font_scale=2)
# sns.set_style("white")

def plot_corr(data, dir=None, filename="kde", color="Greens", show=False):
	# if dir is None:
	# 	raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	dists = pdist(data,metric='correlation')
	corrs = 1-dists
	pylab.figure()
	pylab.hist(corrs,bins=200)
	if dir!=None:
		pylab.savefig("{}/{}.png".format(dir, filename))
	if show:
		pylab.show()
	else:
		pylab.close()

def plot_scatter(x_real, x_fake, dir=None, filename="scatter", show=False):
	# if dir is None:
	# 	raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	real_size = x_real.shape[0]
	data = np.vstack((x_real,x_fake))
	pca = PCA(n_components=2)
	coords = pca.fit_transform(data)
	# tsne = TSNE()
	# coords = tsne.fit_transform(data)
	c_real = coords[:real_size,:]
	c_fake = coords[real_size:,:]
	fig = pylab.figure()
	# fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.scatter(c_real[:, 0], c_real[:, 1], s=20, marker="o", edgecolors="none", color='red',alpha=0.3)
	pylab.scatter(c_fake[:, 0], c_fake[:, 1], s=20, marker="o", edgecolors="none", color='blue',alpha=0.3)
	if dir!= None:
		pylab.savefig("{}/{}.png".format(dir, filename))
	if show:
		pylab.show()
	else:
		pylab.close()
