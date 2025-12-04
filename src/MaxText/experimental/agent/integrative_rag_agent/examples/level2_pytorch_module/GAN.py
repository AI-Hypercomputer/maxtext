import torch.nn as nn

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(128, 256),
        nn.BatchNorm1d(256, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, int(np.prod(img_shape))),
        nn.Tanh()
   )
   def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(int(np.prod(img_shape)), 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    def forward(self, img):
        return self.model(img). 