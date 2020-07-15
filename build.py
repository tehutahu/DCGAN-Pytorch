import torch

from DCGAN.model import Discriminator
from DCGAN.model import Generator

class DCGANBuilder():
    def __init__(self, img_channels, img_size,
                latent_dim=100, pre_trained_D=None, pre_trained_G=None):
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.pre_trained_D = pre_trained_D
        self.pre_trained_G = pre_trained_G

    def _weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def build_discriminator(self):
        D = Discriminator(self.img_channels, self.img_size)
        if self.pre_trained_D:
            D.load_state_dict(torch.load(self.pre_trained_D))
            print("Load Discriminator")
        else:
            D.apply(self._weights_init_normal)
        return D

    def build_generator(self):
        G = Generator(self.img_channels, self.img_size, self.latent_dim)
        if self.pre_trained_G:
            G.load_state_dict(torch.load(self.pre_trained_G))
        else:
            G.apply(self._weights_init_normal)
            print("Load Generator")
        return G

    def build(self):
        D = self.build_discriminator()
        G = self.build_generator()
        return D, G