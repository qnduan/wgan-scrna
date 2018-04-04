from sampler import mat, sample_x, sample_z
import plot
import pylab

zs = sample_z(50,100)
pylab.hist(pylab.ravel(zs),bins=100)

pylab.hist(pylab.ravel(mat[:,:50]),bins=100)

pylab.hist(pylab.ravel(samples_fake_numpy),bins=100)



#plot.plot_corr(mat[:,:100].T)
#
#m1 = mat[:,300:400]
#m2 = mat[:,200:300]
#plot.plot_scatter(m1.T,m2.T)
#
#cc = sample_x(32)
