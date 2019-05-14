
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
    self.l1 = nn.Sequential(nn.Conv2d(3, 10, 8), nn.Conv2d(10, 5, 4), nn.Conv2d(5, 3, 3), nn.MaxPool2d(4), nn.MaxPool2d(3))
    self.encoder_u = nn.Sequential(torch.nn.Linear(3 * 17 * 13, hidden_dim), nn.ReLU())
    self.encoder_v = nn.Sequential(torch.nn.Linear(3 * 17 * 13, hidden_dim), nn.ReLU())
    self.l2 = torch.nn.Linear(hidden_dim, 3 * 17 * 13)
    self.upsample = nn.Sequential(nn.ConvTranspose2d(3, 3, 2, 2), nn.ConvTranspose2d(3, 3, 5, 5))

  def loss(self, reconstructed, x, mu, sigma):
    k = 0
    reconstruction_loss_fn = nn.BCELoss()
    reconstruction_loss = reconstruction_loss_fn(reconstructed, x)
    for i, mu_i in enumerate(mu):
      k -= (1 + torch.log(sigma[i]) - mu_i ** 2 - sigma[i])
    return reconstruction_loss

  def forward(self, x):
    input_size = (3, 218, 178)
    dconv = self.l1(x).view(32, 3 * 17 * 13)

    mu = self.encoder_u(dconv)
    sigma = torch.exp(.5 * self.encoder_v(dconv))
    learned_latent_space = mu + sigma * torch.randn_like(sigma)
    l2 = self.l2(learned_latent_space).view(32, 3, 17, 13)
    upsample = torch.nn.functional.interpolate(input=self.upsample(l2), size=(218, 178),  mode='bilinear')
    reconstructed = nn.Sigmoid()(torch.squeeze(upsample))
    return reconstructed, mu, sigma 

batch_size = 32

img_set = torchvision.datasets.ImageFolder('../data/img', transform=torchvision.transforms.ToTensor(), target_transform=None)

train_loader = torch.utils.data.DataLoader(
                 dataset=img_set,
                 batch_size=batch_size,
                 shuffle=True,
                 pin_memory=False)

vae = VAE(3 * 218 * 178, 500) 

optimizer = torch.optim.Adam(vae.parameters(), lr=3e-4)
epochs = 10
num_batches = len(train_loader)

for epoch in range(epochs):
  for batch_index, sample in enumerate(train_loader):
    loss = 0
    x = sample[0]
    reconstructed, mu, sigma = vae(x)

    loss += vae.loss(reconstructed, x, mu, sigma)

    loss = loss / batch_size
    print(loss.item())

    if batch_index % 50 == 0:
      img_x = smp.toimage(x[0].view(3, 218, 178))
      img_r =  smp.toimage(reconstructed[0].detach().view(3, 218, 178))

      smp.imsave('images/' + str(batch_index) + 'original.jpg', img_x)    
      smp.imsave('images/' + str(batch_index) + 'reconstructed.jpg', img_r)    


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
