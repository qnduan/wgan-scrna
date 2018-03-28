import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import qn

import plot
from model import DisNet, GenNet
import sampler
from progress import Progress

prog = Progress()
config = qn.load('hyperparams.yml')
batchsize = config['batchsize']
dis_net = DisNet(config['dim_x'])
gen_net = GenNet(config['dim_z'],config['dim_x'])

dis_optim = optim.RMSprop(dis_net.parameters(), lr=config['dis_lr'])
gen_optim = optim.Adam(gen_net.parameters(), lr=config['gen_lr'])

for i in range(config['num_updates']):
    for _ in range(config['num_critic']):
        samples_true = sampler.gaussian_mixture_circle(
            batchsize, config['num_mixture'],
            scale=config['scale'],std=config['std'])
        samples_true /= config['scale']
        samples_true = Variable(torch.from_numpy(samples_true))
        z = sampler.sample_z(config['dim_z'],batchsize,
            gaussian=config['gaussian'])
        z = Variable(torch.from_numpy(z))
        samples_fake = gen_net(z).detach()
        samples_fake /= config['scale']

        f_true = dis_net(samples_true)
        f_fake = dis_net(samples_fake)
        loss_critic = f_fake.mean() - f_true.mean()
        prog.add_loss_critic(loss_critic)

        dis_optim.zero_grad()
        loss_critic.backward()
        dis_optim.step()
        dis_net.clip()

    prog.add_loss_dis()
    z = sampler.sample_z(config['dim_z'],batchsize,
        gaussian=config['gaussian'])
    z = Variable(torch.from_numpy(z))
    samples_fake = gen_net(z)
    samples_fake /= config['scale']
    f_fake = dis_net(samples_fake)
    loss_gen = -f_fake.mean()
    prog.add_loss_gen(loss_gen)

    gen_optim.zero_grad()
    loss_gen.backward()
    gen_optim.step()

    if (i+1)%config['num_plot'] == 0:
        print(i+1)
        z = sampler.sample_z(config['dim_z'],10000,
            gaussian=config['gaussian'])
        z = Variable(torch.from_numpy(z))
        samples_fake = gen_net(z).data.numpy()
        plot.plot_scatter(samples_fake,dir='plot',filename='{}_scatter'.format(i+1))
        plot.plot_kde(samples_fake,dir='plot',filename='{}_kde'.format(i+1))
prog.plot()
