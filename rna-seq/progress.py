import numpy as np
import matplotlib.pyplot as plt
import time

class Progress():
    def __init__(self):
        self.loss_dis = []
        self.loss_gen = []
        self.loss_critic = []

    def start(self):
        self.start_time = time.time()

    def duration(self):
        return time.time() - self.start_time

    def add_loss_critic(self,loss):
        self.loss_critic.append(loss)

    def add_loss_gen(self,loss):
        self.loss_gen.append(loss)

    def add_loss_dis(self):
        self.loss_dis.append(np.mean(self.loss_critic))
        self.loss_critic = []

    def plot(self,show=True):
        plt.figure()
        plt.plot(-np.array(self.loss_dis),label='wloss')
        plt.plot(np.array(self.loss_gen),label='gloss')
        plt.legend()
        plt.show()