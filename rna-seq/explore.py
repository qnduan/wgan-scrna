
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

feather.write_dataframe(x,'/Users/qiaonan/Documents/single-cell-seq/explore/gan/mat_6788.ft')


import feather
import numpy as np
from scipy.spatial.distance import pdist
import torch
from model_extract import DisNet, GenNet
from torch import nn
from torch.autograd import Variable
import qn
import pandas as pd

dis_net = DisNet(6982)
dis_params = torch.load('temp_models/dis_net_cpu_10000.pt')
dis_net.load_state_dict(dis_params)

x_ = x.values.T.astype('float32')
y1, y2 = dis_net.forward(torch.from_numpy(x_))
features = y1.data.numpy()
f = pd.DataFrame(features.T)

path = '/Users/qiaonan/Documents/single-cell-seq/explore2/gan'
feather.write_dataframe(f,path+'/pollen_200.ft')
