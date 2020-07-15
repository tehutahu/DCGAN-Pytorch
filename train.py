import os
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

class Trainer():
    def __init__(self, D, G, optimizer_D, optimizer_G,
                 epochs, batch_size, loss, data_loader,
                 out_dir="./output/ver0", sample_interval=50
                 ):
        super(Trainer, self).__init__()
        self.D = D
        self.G = G
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.data_loader = data_loader
        self.sample_interval = sample_interval
        self.latent_dim = G.l1[0].in_features

        gene_dir = os.path.join(out_dir, "generates")
        model_dir = os.path.join(out_dir, "models")
        os.makedirs(gene_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.gene_dir = gene_dir
        self.model_dir = model_dir

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            D.cuda()
            G.cuda()
            self.loss.cuda()
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

    def _train_step(self, n_epoch, iterations, imgs):
        # Adversarial ground truths
        valid = Variable(self.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(self.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()

        # Configure input
        real_imgs = Variable(imgs.type(self.Tensor))
        real_loss = self.loss(self.D(real_imgs), valid)

        # Sample noise as generator input
        z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

        # Generate a batch of images
        gen_imgs = self.G(z)
        fake_loss = self.loss(self.D(gen_imgs), fake)
        real_loss.backward()
        fake_loss.backward()
        self.optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        gen_imgs = self.G(z)
        g_loss = self.loss(self.D(gen_imgs), valid)
        g_loss.backward()
        self.optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (n_epoch, self.epochs, iterations, len(self.data_loader), real_loss.item()+fake_loss.item(), g_loss.item())
        )

    def val_end(self, n_epoch, n_iter):
        n = 25
        with torch.no_grad():
            z = Variable(self.Tensor(np.random.normal(0, 1, (n, self.latent_dim))))
            gen_imgs = self.G(z)
        batches_done = n_epoch * len(self.data_loader) + n_iter
        if batches_done % self.sample_interval == 0:
            file_path = os.path.join(self.gene_dir, "hina_gene%d_%d.png" % (n_epoch, batches_done))
            save_image(gen_imgs.data, file_path, nrow=5, normalize=True)

        if n_epoch % 25 == 0:
            torch.save(self.G.state_dict(), self.model_dir+f"/hina_gene{n_epoch}.pth")
            torch.save(self.D.state_dict(), self.model_dir+f"/hina_dis{n_epoch}.pth")

    def val_step(self, n_epoch, n_iter, imgs):
        self.val_end(n_epoch, n_iter)

    def run(self):
        for epoch in range(self.epochs):
            for i, (imgs, _) in enumerate(self.data_loader):
                self._train_step(epoch, i, imgs)
                self.val_step(epoch, i, imgs)    
        torch.save(self.G.state_dict(), self.model_dir+f"/hina_gene_last.pth")
        torch.save(self.D.state_dict(), self.model_dir+f"/hina_dis_last.pth")