import torch
from torch.autograd import Variable,grad
import numpy as np

x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward(retain_graph=True)


x = Variable(torch.from_numpy(np.array([2])),requires_grad=True)
y = Variable(torch.from_numpy(np.array([3])),requires_grad=True)
z = x*x*y
gradx = grad(z,x,create_graph=True)
gradx[0].backward()

z.backward()


x = Variable(torch.from_numpy(np.array([[2],[3]],dtype='float32')),requires_grad=True)
y = x*x
gradx = grad(y,x,torch.ones(x.size()),create_graph=True)