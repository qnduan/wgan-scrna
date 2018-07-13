
#plot.plot_corr(mat[:,:100].T)
#
#m1 = mat[:,300:400]
#m2 = mat[:,200:300]
#plot.plot_scatter(m1.T,m2.T)
#
#cc = sample_x(32)

import qn

mat = qn.load('data/ipf_log_tpm.pkl')
import pandas as pd
x = pd.DataFrame(mat)

feather.write_dataframe(x,'~/Documents/single-cell-seq/explore/gan/mat_6788.ft')
