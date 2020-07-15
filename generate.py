import argparse
import os
import sys
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from build import DCGANBuilder

out_name = "./sample/sample.png"
out_npy = "./sample/sample.npy"
img_size = 96
latent_dim = 100

def generate(G, batch_size):
    if torch.cuda.is_available():
        G.cuda()
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    with torch.no_grad():
        # G.eval()
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_imgs = G(z)
    save_image(gen_imgs.data, out_name, nrow=int(np.sqrt(batch_size)), normalize=True)
    np.save(out_npy, z.cpu().numpy())
    print("making sample")

def main(args):
    G = DCGANBuilder(3, img_size, latent_dim, pre_trained_G=args.G_path).build_generator()
    generate(G, args.batch_size)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--G_path", type=str, default="./best/hina_gene_last.pth", help="Generater path")
    parser.add_argument("--batch_size", type=int, default=25, help="size of the batches")
    opt = parser.parse_args()
    print(opt)
    return opt

if __name__ == "__main__":
    main(arg_parse())