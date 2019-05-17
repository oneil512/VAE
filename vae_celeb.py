
from __future__ import division
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from torch import nn
import numpy as np
import scipy.misc as smp

class VAE(nn.Module):
  def __init__(self, input_size, hidden_dim):
    super(VAE, self).__init__()
    self.l1 = nn.Sequential(nn.Conv2d(3, 12, 4), nn.Conv2d(12, 6, 4), nn.Conv2d(6, 3, 3), nn.MaxPool2d(2))
    self.encoder_u = nn.Sequential(torch.nn.Linear(3 * 105 * 85, hidden_dim), nn.ReLU())
    self.encoder_v = nn.Sequential(torch.nn.Linear(3 * 105 * 85, hidden_dim), nn.ReLU())
    self.l2 = torch.nn.Linear(hidden_dim, 3 * 105 * 85)
    self.upsample = nn.Sequential(nn.ConvTranspose2d(3, 3, 2, 2))

  def loss(self, reconstructed, x, mu, sigma):
    k = 0
    reconstruction_loss_fn = nn.BCELoss()
    reconstruction_loss = reconstruction_loss_fn(reconstructed, x)
    for i, mu_i in enumerate(mu):
      k -= (1 + torch.log(sigma[i]) - mu_i ** 2 - sigma[i])
    return reconstruction_loss

  def forward(self, x):
    input_size = (3, 218, 178)
    dconv = self.l1(x).view(-1, 3 * 105 * 85)

    mu = self.encoder_u(dconv)
    sigma = torch.exp(.5 * self.encoder_v(dconv))
    learned_latent_space = mu + sigma * torch.randn_like(sigma)
    l2 = self.l2(learned_latent_space).view(-1, 3, 105, 85)
    upsample = self.upsample(l2)
    interpolated = torch.nn.functional.interpolate(input=upsample, size=(218, 178),  mode='nearest')
    reconstructed = nn.Sigmoid()(torch.squeeze(interpolated))
    return reconstructed, mu, sigma, learned_latent_space 

batch_size = 32

img_set = torchvision.datasets.ImageFolder('../data/img', transform=torchvision.transforms.ToTensor(), target_transform=None)

train_loader = torch.utils.data.DataLoader(
                 dataset=img_set,
                 batch_size=batch_size,
                 shuffle=True,
                 pin_memory=False)

vae = VAE(3 * 218 * 178, 500) 

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
epochs = 10
num_batches = len(train_loader)

for epoch in range(epochs):
  for batch_index, sample in enumerate(train_loader):
    loss = 0
    x = sample[0]
    reconstructed, mu, sigma, learned_latent_space = vae(x)

    loss += vae.loss(reconstructed, x, mu, sigma)

    loss = loss / batch_size
    print(loss.item())

    if batch_index % 50 == 0:
      img_x = smp.toimage(x[0].view(3, 218, 178))
      img_r = smp.toimage(reconstructed[0].detach().view(3, 218, 178))

      smp.imsave('images/' + str(batch_index) + '_1original.jpg', img_x)    
      smp.imsave('images/' + str(batch_index) + '_1reconstructed.jpg', img_r)

      img_x = smp.toimage(learned_latent_space[0].detach().view(20,25))
      smp.imsave('images/' + str(batch_index) + '_1latent.jpg', img_x)

      img_x = smp.toimage(x[1].view(3, 218, 178))
      img_r = smp.toimage(reconstructed[1].detach().view(3, 218, 178))

      smp.imsave('images/' + str(batch_index) + '_2original.jpg', img_x)    
      smp.imsave('images/' + str(batch_index) + '_2reconstructed.jpg', img_r)    

      img_x = smp.toimage(learned_latent_space[0].detach().view(20,25))
      smp.imsave('images/' + str(batch_index) + '_2latent.jpg', img_x)




    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
