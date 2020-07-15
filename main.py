import argparse
import os
import sys
import torch
import numpy as np

from data_loader import make_data_loader
from data_loader import make_dataset
from build import DCGANBuilder
from train import Trainer

def main(args):
    img_size = args.img_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    # Load Data
    data_set = make_dataset(img_size, args.image_dir)
    data_loader = make_data_loader(data_set, args.batch_size)
    
    # Make Model
    builder = DCGANBuilder(args.channels, img_size, args.latent_dim,
                           args.load_D, args.load_G
                           )
    D, G = builder.build()

    # Optimizers and Loss
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    adversarial_loss = torch.nn.BCELoss()

    # Train
    runnner = Trainer(D, G, optimizer_D, optimizer_G,
                      args.n_epochs, args.batch_size, adversarial_loss,
                      data_loader, sample_interval=args.sample_interval
                      )
    runnner.run()
    
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=96, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--image_dir", type=str, default="./images", help="image source ROOT directory")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument("--load_D", type=str, help="load D model path (if using)")
    parser.add_argument("--load_G", type=str, help="load G model path (if using)")
    opt = parser.parse_args()
    print(opt)
    return opt

if __name__ == "__main__":
    main(arg_parse())