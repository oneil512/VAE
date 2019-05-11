
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
    self.f1 = nn.Sequential(torch.nn.Linear(input_size, 400), nn.ReLU())
    self.encoder_u = nn.Sequential(torch.nn.Linear(400, hidden_dim), nn.ReLU())
    self.encoder_v = nn.Sequential(torch.nn.Linear(400, hidden_dim), nn.ReLU())
    self.decoder = nn.Sequential(torch.nn.Linear(hidden_dim, input_size), nn.Sigmoid())

  def loss(self, reconstructed, x, mu, sigma):
    k = 0
    reconstruction_loss_fn = nn.BCELoss()
    reconstruction_loss = reconstruction_loss_fn(reconstructed, x)
    for i, mu_i in enumerate(mu):
      k -= (1 + torch.log(sigma[i]) - mu_i ** 2 - sigma[i])
    return reconstruction_loss

  def forward(self, x):
    mu = self.encoder_u(self.f1(x))
    sigma = torch.exp(.5 * self.encoder_v(self.f1(x)))
    learned_latent_space = mu + sigma * torch.randn_like(sigma)
    reconstructed = self.decoder(learned_latent_space)
    return reconstructed, mu, sigma 

batch_size = 32
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True,
                 pin_memory=False)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=False)

vae = VAE(28*28, 200) 

optimizer = torch.optim.Adam(vae.parameters(), lr=3e-4)
epochs = 10
num_batches = len(train_loader)

for epoch in range(epochs):
  for batch_index, sample in enumerate(train_loader):
    loss = 0
    for i, x in enumerate(sample[0]):
      x = x.view(28*28)
      reconstructed, mu, sigma = vae(x)

      loss += vae.loss(reconstructed, x, mu, sigma)

    loss = loss / batch_size
    print(loss.item())

    if batch_index % 25 == 0:
      img_x = smp.toimage(x.view(28, 28))
      img_r =  smp.toimage(reconstructed.detach().view(28, 28))

      img_x.show()    
      img_r.show()


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
